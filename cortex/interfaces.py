"""Shared data structures for Cortex Codegen engines."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ModelStatus(str, Enum):
    DECLARED = "declared"      # defined, parameter-counted, never trained
    TRAINED = "trained"        # checkpoints + evaluation records exist
    UNTRAINED = "untrained"


@dataclass(frozen=True)
class HardwareProfile:
    cpu_cores: int
    ram_total_gb: float
    ram_available_gb: float
    disk_free_gb: float
    has_cuda: bool
    gpu_name: str
    platform: str
    python_version: str
    torch_version: str

    def summary(self) -> str:
        return (
            f"cpus={self.cpu_cores} ram={self.ram_total_gb:.1f}GB "
            f"free_ram={self.ram_available_gb:.1f}GB disk_free={self.disk_free_gb:.1f}GB "
            f"cuda={self.has_cuda} gpu={self.gpu_name}"
        )


@dataclass(frozen=True)
class ModelConfig:
    name: str
    vocab_size: int
    dim: int
    n_layers: int
    n_heads: int
    max_seq_len: int
    ffn_mult: int = 4
    dropout: float = 0.0
    tie_word_embeddings: bool = True
    status: ModelStatus = ModelStatus.UNTRAINED

    @property
    def d_ff(self) -> int:
        return self.ffn_mult * self.dim

    def theoretical_param_count(self) -> int:
        """Exact parameter arithmetic; verified against a live model in tests."""
        d = self.dim
        per_layer = 12 * d * d + 13 * d            # 2 LN + attn(qkvo) + 2 LN + mlp, weights+biases
        emb = self.vocab_size * d
        head = 0 if self.tie_word_embeddings else emb
        return emb + head + 2 * d + self.n_layers * per_layer

    def human_param_count(self) -> str:
        n = self.theoretical_param_count()
        return f"{n:,} ({n / 1e9:.3f}B)" if n >= 1e9 else f"{n:,} ({n / 1e6:.3f}M)"


@dataclass
class Provenance:
    source_uri: str
    license: str
    version: str
    content_sha256: str
    collected_at: str
    collector: str


@dataclass
class KnowledgeItem:
    item_id: str
    kind: str                  # module | class | function | doc | pattern
    language: str
    text: str
    provenance: Provenance
    tags: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "kind": self.kind,
            "language": self.language,
            "text": self.text,
            "provenance": self.provenance.__dict__,
            "tags": self.tags,
        }


@dataclass(frozen=True)
class CapacityTier:
    stored: int
    indexed: int
    retrievable: int
    validated: int


@dataclass(frozen=True)
class ExecutionResult:
    ok: bool
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool
    isolation: str
    limits: dict[str, Any]


@dataclass
class SecurityFinding:
    rule_id: str
    severity: str              # low | medium | high | critical
    file: str
    line: int
    message: str


@dataclass
class PlanStep:
    id: str
    action: str
    target: str
    details: dict[str, Any] = field(default_factory=dict)
    requires_authorization: bool = False


@dataclass
class Plan:
    objective: str
    repo_path: str
    steps: list[PlanStep]
    repository_map: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "repo_path": self.repo_path,
            "steps": [s.__dict__ for s in self.steps],
            "repository_map": self.repository_map,
        }


@dataclass
class ServiceReport:
    engine: str
    status: str                # operational | declared-only | degraded
    evidence: str
    metrics: dict[str, Any] = field(default_factory=dict)
