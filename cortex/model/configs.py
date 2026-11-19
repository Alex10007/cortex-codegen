"""Model configurations and honest status tracking.

The 2B-class configuration is parameter-counted exactly but has never been
trained on this CPU-only host; its status is DECLARED until real checkpoints
and evaluation records exist at that scale.
"""
from __future__ import annotations

from cortex.interfaces import ModelConfig, ModelStatus

CORTEX_DEMO = ModelConfig(
    name="cortex-demo",
    vocab_size=32768,
    dim=192,
    n_layers=4,
    n_heads=3,
    max_seq_len=256,
    ffn_mult=4,
    dropout=0.0,
    tie_word_embeddings=True,
    status=ModelStatus.UNTRAINED,  # set to TRAINED by the training engine on run
)

CORTEX_S = ModelConfig(
    name="cortex-s",
    vocab_size=32768,
    dim=512,
    n_layers=8,
    n_heads=8,
    max_seq_len=1024,
    tie_word_embeddings=True,
    status=ModelStatus.DECLARED,
)

CORTEX_M = ModelConfig(
    name="cortex-m",
    vocab_size=32768,
    dim=1024,
    n_layers=16,
    n_heads=16,
    max_seq_len=2048,
    tie_word_embeddings=True,
    status=ModelStatus.DECLARED,
)

CORTEX_2B = ModelConfig(
    name="cortex-2b",
    vocab_size=32768,
    dim=2560,
    n_layers=24,
    n_heads=20,
    max_seq_len=8192,
    ffn_mult=4,
    dropout=0.0,
    tie_word_embeddings=False,
    status=ModelStatus.DECLARED,
)

CONFIGS: dict[str, ModelConfig] = {
    c.name: c for c in (CORTEX_DEMO, CORTEX_S, CORTEX_M, CORTEX_2B)
}

# Status ledger. The training engine writes trained-ness here after real runs;
# nothing else may promote a status.
MODEL_STATUS: dict[str, dict] = {
    "cortex-demo": {"status": "untrained", "checkpoint": None, "eval": None},
    "cortex-s": {"status": "declared", "checkpoint": None, "eval": None},
    "cortex-m": {"status": "declared", "checkpoint": None, "eval": None},
    "cortex-2b": {
        "status": "declared",
        "checkpoint": None,
        "eval": None,
        "note": (
            "2B-class physical parameter target (>=2e9 params, exact count from "
            "ModelConfig.theoretical_param_count). Not trained on the CPU-only "
            "development host; training path is provided for GPU scale-out."
        ),
    },
}


def get_config(name: str) -> ModelConfig:
    return CONFIGS[name]


def declared_configs() -> dict[str, int]:
    return {name: cfg.theoretical_param_count() for name, cfg in CONFIGS.items()}


def record_training_run(name: str, checkpoint_path: str, eval_path: str | None) -> None:
    cfg = CONFIGS[name]
    if cfg.status is ModelStatus.DECLARED:
        raise PermissionError(
            f"{name} is a declared-scale config; promotion to trained requires a "
            "host with sufficient compute. Only the demo config may be trained here."
        )
    MODEL_STATUS[name] = {
        "status": "trained",
        "checkpoint": checkpoint_path,
        "eval": eval_path,
    }
