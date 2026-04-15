# Issue: KVzapConfig.__init__ crashes when called with no arguments

## Title
`KVzapConfig.__init__` missing default values breaks `to_diff_dict()`

## Description

`KVzapConfig.__init__` requires `input_dim`, `output_dim`, and `n_modules` as mandatory keyword-only parameters:

```python
class KVzapConfig(PretrainedConfig):
    model_type = "kvzap"

    def __init__(self, *, input_dim: int, output_dim: int, n_modules: int,
                 hidden_dim: Optional[int] = None, **kwargs):
        ...
```

`transformers.PretrainedConfig.to_diff_dict()` internally calls `self.__class__()` with no arguments to obtain a default config for comparison. This raises:

```
TypeError: KVzapConfig.__init__() missing 3 required keyword-only arguments:
    'input_dim', 'output_dim', and 'n_modules'
```

### Reproduction

```python
from kvpress import KVzapPress

press = KVzapPress(model_type="mlp")
press.kvzap_model.config.to_diff_dict()
# TypeError
```

This crashes any code path that serializes the config, including `save_pretrained()` and logging/debugging flows.

### Suggested Fix

Make all dimension parameters optional with `None` defaults:

```python
def __init__(
    self,
    *,
    input_dim: Optional[int] = None,
    output_dim: Optional[int] = None,
    n_modules: Optional[int] = None,
    hidden_dim: Optional[int] = None,
    **kwargs,
):
```

This is safe because:
1. `KVzapModel.__init__` reads these from the config *after* `from_pretrained()` sets them — defaults are never used during model construction
2. Matches the pattern used by other `PretrainedConfig` subclasses (`BertConfig`, `GPT2Config`, etc.)
3. Backward compatible — existing code passing explicit values continues to work
4. Published pretrained weights (HF hub) remain reproducible since they always provide explicit dimensions

### Context

Encountered during RULER leaderboard evaluation when using `KVzapPress` as a scorer inside a composition wrapper. The crash occurred during results serialization.

---

*🤖🤖🤖 This issue was prepared with AI assistance.*
