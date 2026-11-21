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


## Stage 17E — Strict execution-evidence learning path

The existing testing, pipeline, core, and learning components are now connected
through a strict path: API-triggered engineering pipelines request full
Bubblewrap plus delegated cgroup isolation; test reports retain kernel/resource
evidence; the existing LearningEngine accepts pipeline outcomes only when
implementation, objective verification, tests, security, and required cgroup
controls all pass. Verified Python changes can then enter the existing
provenance-aware knowledge store. Unverified outcomes are rejected.

This is verified knowledge ingestion, **not** online gradient updates,
continual model-weight learning, or a claim that the model learned from these
examples. The path is opt-in through the authenticated API pipeline; it has not
been runtime-qualified on the current branch.


## Stage 17F — Graph-backed change-impact analysis

The existing `KnowledgeGraph` now exposes bounded reverse-dependency impact
analysis for changed repository paths. It resolves source paths through
existing node provenance, expands changed modules to their defined symbols,
and traverses incoming local-import, call, and inheritance edges. Results
include impacted node IDs, paths, relationship types, depth, unresolved paths,
and truncation state. Path traversal is constrained to the supplied repository
root and the traversal has explicit depth/node ceilings.

`CortexAgent.analyze_change_impact` exposes this capability through the
existing facade. This is not yet automatic selective-test execution: the
engineering pipeline still runs its existing test command, and multi-language
symbol resolution remains incomplete. Regression tests cover call/import/
inheritance impact and path/bounds rejection; execution remains pending.


## Stage 17G — Authenticated change-impact API

The existing authenticated FastAPI surface now exposes `POST /repo/impact`.
It accepts a repository path, a bounded list of changed paths, and explicit
depth/node limits; repository access is constrained by the existing
`CORTEX_ALLOWED_REPO_ROOTS` policy and the endpoint uses the existing
expensive-operation rate limiter. It delegates to the canonical
`CortexAgent.analyze_change_impact` method and returns the graph's unresolved
paths rather than guessing when source provenance is missing.

This endpoint reports impact; it does not modify files, run tests, or claim
that affected tests were selected automatically. API tests and runtime
qualification remain pending.


## Stage 17H — Conservative regression-test recommendation

The existing `cortex.testing.engine` now maps changed and impacted Python
modules to likely pytest files using import relationships and test naming.
The `CortexAgent.analyze_change_impact` response includes this recommendation.
If no confident match exists, it explicitly falls back to the discovered full
test suite. Parse failures, truncation, and limitations are reported.

This is static test-target recommendation, not dependency-complete coverage
analysis. The engineering pipeline continues to run its full configured suite;
no tests are silently skipped and no release gate has been weakened.
Regression-selector tests were added but not executed.


### Stage 17H hardening note

Regression-test discovery now prunes common generated/vendor directories,
caps discovery at 10,000 test files, includes unparseable test files
conservatively in targeted recommendations, and treats discovery truncation
as a reason to fall back to the full discovered suite. Changed-path inputs are
bounded to 4,096 characters. The static capability audit now checks both the
impact endpoint and the existing adaptive-generation API in one canonical
API-file entry.


### Stage 17D retrieval-context formatting correction

Corrected escaped newline literals in the retrieval-context formatter so
verified facts, hierarchical events, and evidence sections are separated by
actual line breaks in the model prompt. No stored memory records or facts were
modified.


## Stage 17I — Engineering-pipeline change-impact integration

The existing CortexAgent.run_pipeline now wraps the caller's existing patcher at the integration boundary. When a non-empty edit set is proposed, it passes the changed paths to the canonical analyze_change_impact facade before the pipeline publishes edits. The resulting impact report and conservative regression-test recommendation are attached to the pipeline result and summarized in the existing audit log.

The recommendation is advisory only. The existing configured test command and full-suite behavior are unchanged; the pipeline does not silently replace full-suite verification with a partial targeted run. If graph analysis fails, the pipeline records an explicit change_impact.status=unavailable result and continues under the unchanged test/security verification policy. No second impact engine, dependency graph, or test selector was introduced.

Added integration tests for successful report propagation, audit metadata, unchanged full-suite command, and impact-analysis failure behavior. These tests are committed but have not yet been executed. The implementation does not claim complete dependency coverage, automatic test selection, or runtime qualification.


## Stage 17J — Cross-language repository symbol inventory

Extended the existing `cortex.knowledge.indexer` to add syntax-level module and declaration nodes for JavaScript (`.js`, `.jsx`), TypeScript (`.ts`, `.tsx`), Rust (`.rs`), Go (`.go`), and Java (`.java`) to the canonical SQLite knowledge graph. It records language-qualified module IDs and declaration types, and creates `defines` edges. Relative JavaScript/TypeScript imports and simple Rust module declarations are resolved only when the destination maps to one indexed source file; unresolved or ambiguous references are counted rather than guessed. Python AST indexing remains unchanged and no second graph or index was introduced.

Important limitation: these additional language scanners use conservative regular expressions, not compiler-grade parsers. They do not provide complete scope resolution, call-graph construction, type inference, macro expansion, package-manager resolution, Java package/import resolution, or Go package dependency resolution. Comments and string literals can cause false declaration matches. The new counts distinguish foreign-language files/symbols/imports from existing Python counts. These are repository-navigation foundations, not complete multi-language semantic intelligence.

Added tests covering symbol indexing across TypeScript, Rust, Go, and Java, unique relative TypeScript import resolution, and explicit unresolved-import accounting. Tests are committed but have not been executed; the configured GitHub Actions runner has not yet produced a run for the current branch head. No training, checkpoint changes, or capability promotion occurred.



## Stage 17K — Dependency-cycle and architecture-boundary diagnostics

Extended the existing SQLite-backed `KnowledgeGraph` with two bounded
repository diagnostics and exposed them through
`CortexAgent.analyze_architecture`:

- `dependency_cycles` computes strongly connected components over selected
  canonical graph relationships (imports-local by default). It reports
  cyclic components, their provenance-backed repository paths, graph bounds,
  and truncation.
- `architecture_boundary_violations` evaluates caller-supplied
  repository-relative path-prefix rules against selected graph edges
  (imports-local by default). It reports the exact source/target paths,
  nodes, relationship and rule index. Prefix matching is segment-aware, so
  `domain2/` does not match a `domain/` rule.
- Both operate on the existing graph and only include nodes with
  provenance-resolvable paths inside the requested repository root. The
  implementation uses iterative strongly connected component traversal to
  avoid Python recursion-depth dependence.

Architecture rules are explicit inputs; Cortex does not invent a project's
layering policy or claim to infer architecture intent. Results are diagnostic
and advisory. They are not wired as a release gate, do not modify files, and
do not change the engineering pipeline's full-suite test behavior.

Limitations: cycles are only as complete as indexed relationships; unresolved
imports, missing graph nodes, unindexed languages and dynamic dependencies
remain invisible. Strongly connected components can contain multiple simple
cycles. Boundary checks only evaluate the selected edge types and supplied
path-prefix rules. Tests were added for multi-node cycles, self-loops,
segment-aware boundary matching, and path-traversal rejection, but have not
been executed. No model training, checkpoint change, or capability promotion
occurred.



## Stage 17L — Scope-aware Python symbol resolution and type relationships

Extended the existing Python AST indexer and canonical knowledge graph with
lexical-scope-aware resolution. Symbol declarations are keyed by module,
lexical scope, and name. Calls are resolved only against declarations visible
in the current lexical scope chain. Explicit `self.method()` and
`cls.method()` references are resolved against the nearest enclosing class.
Local parameters, assignments, and imports are treated as shadowing bindings;
same-scope duplicate declarations are reported as ambiguous rather than
collapsed into a false unique match.

Python function argument, return, and annotated-assignment type names are
connected to uniquely resolved local class/type declarations through
`type_uses` edges. Existing `calls` and `inherits` edge types remain
canonical and unchanged. New counts distinguish ambiguous calls,
ambiguous inheritance/type references, and resolved type references.

Scope of this increment: the new semantic resolution is Python-AST-based.
It does not yet resolve imported aliases across modules, dynamic attributes,
descriptor/metaclass behavior, runtime monkey-patching, generic type aliases
with full typing semantics, or semantic relationships in the regex-indexed
JavaScript/TypeScript/Rust/Go/Java inventories. Unresolved references are
not guessed. This is static lexical resolution, not compiler-grade whole-
program analysis or proof of runtime dispatch.

Added tests for nested lexical calls, instance-method calls, parameter
shadowing, duplicate-name ambiguity, and Python type-annotation edges.
Tests are committed but have not been executed; current-head CI has not
provided a runtime result. No model training, checkpoint changes, benchmark
promotion, or Stage 12 champion change occurred.
