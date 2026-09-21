# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


def search_hyperplane(X, max_iter: int = 1000):
    """
    Given a tensor X of shape (bsz, seq_len, head_dim), search for a hyperplane Y (bsz, head_dim)
    such that for every i, <X[:, i], Y> <= 0. Returns -1e5 * Y / ||Y||², uniformly scaled down when
    necessary to keep it representable in the input dtype.

    Parameters
    ----------
    X : torch.Tensor
        Query tensor with shape (batch_size, seq_len, head_dim) representing
        the query vectors for which we want to find a nullifying hyperplane.
    max_iter : int, default=1000
        Maximum number of iterations to search for the hyperplane. If no valid
        hyperplane is found within this limit, a ValueError is raised.

    Returns
    -------
    torch.Tensor
        Hyperplane tensor with shape (batch_size, head_dim) and the same dtype as X.

    Raises
    ------
    ValueError
        If no valid hyperplane is found within max_iter iterations.
    """
    output_dtype = X.dtype
    # float16 can overflow during the search and while constructing the fake key.
    if output_dtype == torch.float16:
        X = X.float()

    Y = X.mean(1)  # this initialization is enough for most cases
    for _ in range(max_iter):
        mask = torch.bmm(X, Y.unsqueeze(-1)) <= 0
        if not mask.any():
            K = -1e5 * Y / Y.norm(dim=-1, keepdim=True) ** 2
            if output_dtype == torch.float16:
                scale = (torch.finfo(output_dtype).max / K.abs().amax(dim=-1, keepdim=True)).clamp(max=1)
                K = (K * scale).to(output_dtype)
            return K
        Y += (X * mask).sum(1) / mask.sum(1).clamp(min=1)
    raise ValueError("Could not find fake keys such that for every query q, exp(<q, k>) = 0")


def apply_merge_logit_bias(module, query, key, attention_mask, bias):
    """
    Add a per-key additive logit bias (``module.merge_logit_bias``, shape ``(bsz, num_kv_heads, n_cached)``)
    to the attention mask. Used by :class:`~kvpress.presses.merging_press.MergingPress`: with
    ``count_logit_bias=True`` a survivor that absorbed ``M`` evicted tokens carries ``log(1 + M)``
    (KeepKV, arXiv:2504.09936), so its softmax odds are multiplied by ``1 + M``; with ``bias="score"``
    it carries ``log(1 + sum_j exp(s_j - s_i))``, the attention mass its folded tokens held under the
    press score, so the compressed softmax assigns the survivor the mass of itself and its fold.
    The bias tensor is indexed in cache order: full length for mask-based inner presses, survivor
    order (``n_kept``) for the truncated ScorerPress path.

    Cached positions beyond ``bias.shape[2]`` (new tokens) get bias 0. The bias is broadcast from
    key-value heads to query heads (``repeat_interleave``, the ``repeat_kv`` order). If
    ``attention_mask`` is ``None`` a causal float mask is built first, so the returned mask is always
    a 4D additive float mask of shape ``(bsz, num_heads, q_len, k_len)`` in the query dtype.

    Returns ``attention_mask`` unchanged when the bias is identically zero.
    """
    if bias is None or not bool(bias.any()):
        return attention_mask
    bsz, num_heads, q_len, _ = query.shape
    num_kv_heads, k_len = key.shape[1], key.shape[2]
    if bias.shape[2] > k_len:
        raise ValueError(f"merge_logit_bias covers {bias.shape[2]} positions but the cache has {k_len}")
    b = bias.to(device=query.device, dtype=query.dtype)
    if b.shape[2] < k_len:
        b = torch.nn.functional.pad(b, (0, k_len - b.shape[2]))
    b = b.repeat_interleave(num_heads // num_kv_heads, dim=1).unsqueeze(2)  # (bsz, num_heads, 1, k_len)

    min_val = torch.finfo(query.dtype).min
    if attention_mask is None:
        if q_len > 1:
            kv_idx = torch.arange(k_len, device=query.device)
            q_idx = torch.arange(q_len, device=query.device) + (k_len - q_len)
            allowed = kv_idx[None, :] <= q_idx[:, None]
            fmask = torch.zeros(q_len, k_len, dtype=query.dtype, device=query.device)
            fmask = fmask.masked_fill(~allowed, min_val)[None, None]
        else:
            fmask = torch.zeros(1, 1, 1, k_len, dtype=query.dtype, device=query.device)
    elif attention_mask.dtype == torch.bool:
        fmask = torch.zeros(attention_mask.shape, dtype=query.dtype, device=query.device)
        fmask = fmask.masked_fill(~attention_mask, min_val)
    else:
        fmask = attention_mask.to(query.dtype)
    if fmask.ndim == 4:
        fmask = fmask[:, :, :, :k_len]
    module.merge_bias_calls = getattr(module, "merge_bias_calls", 0) + 1
    return fmask + b


def attention_patch(func):
    """
    Decorator to update the keys before the attention computation at the indices provided in module.masked_key_indices
    The keys are updated with a fake key k whose scaled attention weight underflows to zero
    This solution is not optimal as it does not reduce peak memory and slightly increases runtime

    Parameters
    ----------
    func : callable
        The original attention function to be patched. Should accept parameters
        (module, query, key, value, attention_mask, dropout, **kwargs).

    Returns
    -------
    callable
        The wrapped attention function that supports head-wise key masking.
    """

    def wrapper(module, query, key, value, attention_mask, dropout, **kwargs):
        if query.shape[2] == key.shape[2]:
            # Prefilling
            module.masked_key_indices = None
            module.merge_logit_bias = None
        elif getattr(module, "merge_logit_bias", None) is not None:
            # Decoding with merged survivors (MergingPress count_logit_bias / bias="score"): additive per-key logit bias
            attention_mask = apply_merge_logit_bias(module, query, key, attention_mask, module.merge_logit_bias)
        if query.shape[2] != key.shape[2] and getattr(module, "masked_key_indices", None) is not None:
            # Decoding: build fake keys k s.t. exp(<q, k>) = 0
            bsz, num_heads, seq_len, head_dim = query.shape
            num_key_value_heads = key.shape[1]
            num_groups = num_heads // num_key_value_heads

            # Build a fake key k per key group such that for every query q, exp(<q, k>) = 0
            q = query.view(bsz, num_key_value_heads, num_groups, seq_len, head_dim)
            q = q.reshape(bsz * num_key_value_heads, num_groups * seq_len, head_dim)
            k = search_hyperplane(q)
            k = k.view(bsz, num_key_value_heads, head_dim)

            # At indices, update the keys to the fake keys
            batch_indices, head_indices, seq_indices = module.masked_key_indices
            key[batch_indices, head_indices, seq_indices] = k[batch_indices, head_indices]

        # see https://github.com/NVIDIA/kvpress/pull/115#issuecomment-3183785597
        # cu_seq_lens_k are only in kwargs if model.generate is used.
        if "cu_seq_lens_k" in kwargs:
            kwargs["cu_seq_lens_k"][-1] = key.shape[-2]
        return func(module, query, key, value, attention_mask, dropout, **kwargs)

    return wrapper


def patch_attention_functions():
    """
    Apply attention patching to all transformer attention functions.

    This function automatically patches all attention functions registered in
    transformers' ALL_ATTENTION_FUNCTIONS to support head-wise key masking.
    It enables KVPress compression methods that require head-specific masking
    (like AdaKV) to work correctly during text generation.

    The patching is applied globally and affects all transformer models loaded
    after this function is called. It's automatically called when importing
    kvpress to ensure compatibility with head-wise compression methods.

    Notes
    -----
    This function modifies the global attention functions in the transformers
    library. The modifications do not affect models that don't use head-wise compression (i.e. don't have
    module.masked_key_indices).
    """
    for name, func in ALL_ATTENTION_FUNCTIONS.items():
        ALL_ATTENTION_FUNCTIONS[name] = attention_patch(func)
