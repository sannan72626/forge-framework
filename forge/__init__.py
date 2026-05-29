"""FORGE: Fisher-Orchestrated Robust Gating for Erasure."""

BASE_MODEL = "outputs/wk0/phi3_tofu_full"
HIDDEN_DIM = 3072
MAX_LEN    = 128

try:
    from .trainer     import FORGETrainer, FORGEArgs, forge_train_standalone
    from .adaptive_lr import FORGEScheduler
except ImportError:
    pass  # torch not yet installed

try:
    from .metrics import ForgeEvaluator
except ImportError:
    pass
