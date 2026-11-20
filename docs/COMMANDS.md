# Evaluation commands

Run these modules from the checkout after installation. All study data and
completed results are external inputs. No command below generates DELTA banks,
counterfactual patches or correction operators.

## Installation

```bash
python -m pip install -e ".[test,reporting]"
python -m pytest -q -m "not models"
```

The `models` extra adds PyTorch for MIL, Cox and exact normative k-nearest-neighbour
evaluation. The `benchmark` extra additionally provides InMoose for ComBat. Choose
a system-compatible PyTorch build. The `molecular` extra provides expression-data
readers and the Torch/TorchMetrics runtime required to reconstruct section-level
molecular metrics. None of these groups supplies the unreleased DELTA module.

## Interfaces

| Evaluation | Command |
|---|---|
| Patch-level assessment | `python -m experiments.pathorob.run --help` |
| Patch numerical report | `python -m experiments.pathorob.report --help` |
| Patch paired summaries | `python -m experiments.pathorob.summary --help` |
| Patch workbook export | `python -m experiments.pathorob.workbook --help` |
| Evaluation partitions | `python -m experiments.common.split_cli --help` |
| Aggregation report | `python -m experiments.patch_aggregation.report --help` |
| Aggregation paired summaries | `python -m experiments.patch_aggregation.summary --help` |
| Survival report | `python -m experiments.wsi_survival.report --help` |
| Survival workbook merge | `python -m experiments.wsi_survival.merge --help` |
| Survival paired summaries | `python -m experiments.wsi_survival.summary --help` |
| Coverage-result report | `python -m experiments.bank_coverage.report --help` |
| Screening report | `python -m experiments.humc_normative.report --help` |
| Screening paired summaries | `python -m experiments.humc_normative.summary --help` |
| Stain/scanner result collection | `python -m experiments.plism_analysis.collect --help` |
| Stain/scanner report | `python -m experiments.plism_analysis.report --help` |
| Supplied stain/scanner embeddings | `python -m experiments.plism_analysis.evaluate --help` |
| Molecular result audit | `python -m experiments.stpbench_molecular_prediction.report --help` |
| Molecular paired inference | `python -m experiments.stpbench_molecular_prediction.statistics --help` |
| Molecular patient-cluster inference | `python -m experiments.stpbench_molecular_prediction.external_statistics --help` |
| Molecular workbook export | `python -m experiments.stpbench_molecular_prediction.tables --help` |

## Numerical report examples

Supply the corresponding completed inputs and choose a new output directory.
The paths below are examples, not files included in the checkout.

```bash
python -m experiments.pathorob.summary \
  --input /path/to/patch_results_snapshot.json \
  --output-dir outputs/patch_summary

python -m experiments.patch_aggregation.report \
  --input-csv /path/to/aggregation_results_long.csv \
  --output-dir outputs/aggregation_report

python -m experiments.plism_analysis.evaluate \
  --input /path/to/corresponding_features.npz \
  --output outputs/stain_scanner_metrics.csv

python -m experiments.stpbench_molecular_prediction.external_statistics \
  --input /path/to/section_metrics.csv \
  --output-dir outputs/molecular_external_statistics
```

The feature-level PLISM input contains `raw_reference`, `raw_query`,
`adjusted_reference`, `adjusted_query` and `tissue_labels`. Rows identify
corresponding tissue regions. Adjusted embeddings must have been supplied by an
upstream process; this command performs no correction. The molecular input
contains `cohort`, `panel`, `patient` and paired `pcc_raw`, `pcc_delta`,
`mae_raw`, `mae_delta` columns. The HUMC report requires an externally supplied
case-level color audit through `report --color-audit`, alongside its completed
predictions and output root.

The [input guide](INPUTS.md) distinguishes supplied predictions and representations
from the withheld correction stage. Actual result tables must preserve their
source identities, comparison methods and statistical units.
