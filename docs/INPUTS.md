# Evaluation inputs

This repository evaluates supplied representations and predictions. The DELTA
module is not included. Bank construction, counterfactual generation, operator
estimation, projection and foundation-model feature extraction remain outside
this release. End-to-end reproduction requires those upstream stages and the
matching study data.

## Retained modules

| Evaluation | Modules | External inputs |
|---|---|---|
| Patch representation and classification | `experiments.pathorob.evaluation`, `run`, `report`, `summary`, `workbook` | Embeddings with row-aligned patch metadata, evaluation partitions and completed per-method results |
| Patch-aggregated classification | `experiments.patch_aggregation.evaluation`, `report`, `statistics`, `summary` | Per-method patch embeddings, slide/patient membership, labels, partitions or completed long-format results |
| WSI-level survival | `experiments.wsi_survival.model`, `metrics`, `report`, `merge`, `summary` | Patient representations, survival time/event records, partitions or completed endpoint tables |
| HUMC normative screening | `experiments.humc_normative.scoring`, `report`, `summary` | Supplied per-method embeddings, reference/held-out case membership or completed case predictions |
| Staining and scanner variation | `experiments.plism_analysis.evaluate`, `statistics`, `report`, `collect` | Externally supplied raw/adjusted embeddings or completed per-condition measurements with corresponding tissue-region and acquisition identifiers |
| Spatial molecular prediction | `experiments.stpbench_molecular_prediction.report`, `statistics`, `external_statistics`, `tables` | Saved gene-expression predictions, aligned measured expression, section/patient identifiers and evaluation configuration |
| Bank coverage results | `experiments.bank_coverage.report` | Completed per-fraction endpoint tables and their matched reference results |

The common utilities resolve external paths, identify source files and construct
evaluation partitions. They do not estimate or apply a DELTA operator. Retained
ComBat and downstream learner code belongs to the comparison and evaluation
stages.

Patch representation endpoints additionally require the declared upstream
PathoROB metric implementation. The patch evaluator accepts that external
checkout explicitly; it is not bundled here. Molecular section-level metric
reconstruction requires the `molecular` dependency group and the matching
prepared expression assets. Its external patient-cluster report accepts
completed section metrics directly. Exact external sign enumeration is bounded
to 23 patient clusters per task, matching the retained protocol.

## Input identity and statistical units

Keep row order, representation branch, encoder identity, cohort membership and
partitions consistent across methods. Correction must have been completed
upstream with the declared training scope. An evaluator cannot establish that
scope from an embedding matrix alone.

Patch results retain the benchmark's official partition identities. Aggregation
results identify the slide or patient and the relevant task. Survival results
retain patient-level events and follow-up times. Screening inference retains
case identifiers and paired backbone results. Molecular comparisons retain the
section and, where specified, the patient cluster. Statistical units and
comparison families are preserved by the task-specific reporting modules.

HUMC images, identifiers, labels, fold assignments, embeddings and results are
restricted institutional data. External transfer is prohibited. The corresponding
evaluation code is retained for authorized execution within the institution and
does not provide access to the data.

## Data and software checks

No actual study images, embeddings, cohort tables, labels, partitions, reference
statistics, fitted operators or result snapshots are distributed. Paths in command
examples refer to independently supplied inputs. Public dataset access remains
subject to the source owners' terms. The backbone registry contains public model
identities and extraction specifications, not model weights or features.

Tests construct synthetic inputs and check genuine evaluation formulas,
statistical inference and CSV/Excel serialization. They do not substitute for
the withheld DELTA implementation or demonstrate clinical performance.
