# Cortex Stage 17 — Unified Intelligence Foundation

Owner: **Fibonera**  
Branch: `vibe/cortex-core-b71b2c`  
Parent: Stage 16 integrated branch head  
Status: **IMPLEMENTED IN SOURCE; RUNTIME TESTS NOT YET EXECUTED**

## Objective

Introduce a non-duplicative foundation for metacognitive task assessment and
hierarchical persistent memory while preserving the existing planner, memory
database, knowledge graph, MoE, divergent evaluation, execution sandbox,
training system, and all historical artifacts.

## Implemented in this increment

### 1. Metacognitive task assessment — foundation

Extended `cortex/planning/engine.py` with `assess_objective`, called by the
existing `plan` function. It produces explicit ambiguity, complexity,
uncertainty, risk, reasoning-depth, verification-profile, evidence, and
authorization recommendations.

Important limitation: this is a deterministic policy heuristic, **not** a
learned metacognitive model, calibrated confidence estimate, or proof of
reasoning quality. It does not itself increase model intelligence or execute
additional reasoning passes.

The existing Plan structure is extended additively with an assessment field.
Existing step ordering is retained for non-destructive objectives.

### 2. Hierarchical persistent memory — foundation

Extended the existing `MemoryEngine` and its SQLite schema with additive
migrations. Event records can be classified as episodic, semantic, procedural,
or decision memory, with confidence and source metadata.

Added append-only fact history. A conflicting unverified assertion cannot
overwrite an active verified fact; it is retained as contested history.
Verified revisions preserve superseded values for provenance and audit.

No second memory database, retrieval engine, or vector store was introduced.
Existing method defaults remain backward-compatible.

### 3. Capability ownership map

Added `scripts/stage17_audit.py` and
`artifacts/stage17_capability_inventory.json` generation. The inventory maps
the 15 strategic capabilities to canonical existing modules and labels each
as foundation-implemented, partial, or not implemented. Static presence is
not runtime or capability qualification.

## Tests added

`tests/test_stage17.py` covers:
- explicit heuristic semantics and risk classification;
- ambiguous task assessment and authorization recommendations;
- plan serialization compatibility;
- memory tiers, confidence validation and summary counts;
- verified-fact contradiction protection and historical revisions;
- additive migration from the legacy SQLite schema.

## Not claimed complete

Recursive general-purpose reasoning, model-learned metacognition, full
cross-language repository graphs, model-directed autonomous patching,
online execution-grounded model learning, formal proof, continual learning,
autonomous curriculum generation, actual adaptive compute scheduling,
mathematical/scientific reasoning, multimodal engineering, and game/simulation
qualification remain future implementation work.

## Preservation and qualification

- Stage 12 champion: unchanged.
- Historical Stage 7–16 artifacts: unchanged.
- Model weights/checkpoints: unchanged.
- Physical/virtual training: not performed.
- GPU qualification: not performed.
- Runtime test suite: not executed in this environment.
- Sandbox isolation: not newly qualified.
- Capability promotion: none.

The next increment should extend the existing knowledge indexer and graph into
a repository-wide symbol/dependency model, then connect its impact analysis to
the existing planner and verification pipeline.


## Stage 17B — Repository-local dependency and symbol graph

The existing `cortex.knowledge.indexer` now adds typed local-import,
function-call, and class-inheritance edges to the existing SQLite knowledge
graph. Resolution is conservative: call/inheritance edges are emitted only
when a local symbol name resolves uniquely. Ambiguous and external calls are
counted as unresolved rather than guessed. The existing vector store and
graph remain the canonical stores; no second repository index was created.

Tests were added for local import, call, and inheritance relationships.
These tests are committed but have **not yet been executed** because the
configured GitHub Actions runner has not been shown to execute workflow steps.
The graph remains Python-AST-based; full TypeScript, JavaScript, Rust, Go and
cross-language semantic resolution is still outstanding.


## Stage 17C — Adaptive divergent-evaluation budget

The existing divergent evaluation can now opt into `adaptive_budget=True`.
It reuses the planner's deterministic task assessment to select one, two, or
three candidate hypotheses per task. High/critical-risk tasks receive at least
two hypotheses; deeper tasks may receive three. The default remains the
historical fixed three-hypothesis behavior, preserving benchmark comparability.

This is adaptive candidate-count allocation, **not** dynamic neural-network
compute scaling, learned confidence, or recursive general-purpose reasoning.
A regression test verifies dispatch counts without invoking model inference;
runtime model-quality improvement has not been measured.


## Stage 17D — Memory-aware retrieval and bounded compute allocation

The existing Cortex retrieval context now consumes verified memory through
tier-prioritized retrieval (procedural, semantic, decision, episodic), and
includes confidence/source/revision metadata for verified facts. Unverified
events remain excluded from trusted retrieval.

Generation now exposes opt-in `adaptive_compute`. It uses the existing
assessment policy to cap output tokens at 50%, 75%, or 100% of the caller's
requested maximum, while never exceeding the requested cap. The API exposes
the same opt-in field. This is bounded output-token allocation, not dynamic
GPU scheduling, adaptive attention depth, or learned compute control.
