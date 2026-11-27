# Paper figures — CSDS463 MIL fusion project

This folder contains six vector **SVG figures** tracked directly in GitHub. **The original 300-dpi PNG and vector PDF exports are not in GitHub yet**; they are in the separate figure pack generated in ChatGPT and can be copied into this folder before a normal `git add/commit/push` from Pioneer or another authenticated Git client.

## Figure index

| Figure | File | Section | Scope |
|---|---|---|---|
| 1 | [Experimental workflow](figure_01_workflow.svg) | Methods | Study design |
| 2 | [Held-out methods](figure_02_heldout_methods.svg) | Results | Held-out test ROC AUC, 3 seeds |
| 3 | [Stability trade-off](figure_03_stability_tradeoff.svg) | Results/discussion | Held-out mean versus across-seed SD |
| 4 | [Paired seed effects](figure_04_seed_paired_test_changes.svg) | Results/supplement | Held-out paired seed changes |
| 5 | [Experiment A](figure_05_experiment_a_ensembles.svg) | Exploratory results | Validation only |
| 6 | [Experiment B](figure_06_experiment_b_greedy_soups.svg) | Exploratory results | Validation only |

## Provenance and interpretation

- Figures 2–4: existing `final_results/three_way_summary.csv`, `final_results/summary.csv`, and `final_results/all_results.csv`. These were already committed from previous CLAM/DSMIL evaluations. The test split has 128 slides. Error bars are **sample SD across only three seeds**, never confidence intervals.
- Figure 5: `experiment_a_outputs/validation_auc_results.csv` created on Pioneer from the completed array job `4318825`. The SVG is based on the six-decimal stdout measurements supplied in the project conversation, so it is a visualization snapshot; generate final PNG/PDF from the **on-cluster CSV** to avoid rounding differences.
- Figure 6: `final_results/experiment_b_summary.csv` created on Pioneer from all nine manifests of job `4318839`. SVG points are based on the six-decimal values supplied in the project conversation. CLAM seed 1 improved from 0.794034 to 0.796875; the other eight soups selected a single checkpoint.
- Figure 1: methods schematic, not numerical data. All architectures were not evaluated with precisely identical training/continuation protocols.

**Do not interpret A or B plots as held-out test outcomes.** Experiment B selected checkpoints on the same validation slides used in its AUC comparison. The held-out analyses have only three seeds and their sample standard deviations are descriptive.

**Outstanding figures:** Weighted TIES variants (AUC-weighted versus stability-weighted) have completed training but require a verified results table before plotting. ABMIL held-out exists on Pioneer but is not included in Figure 2; avoid adding invented values.

## Generating local PNG/PDF exports

Use the tracked plotting script from the repository root on Pioneer:

```bash
python paper_figures/plot_from_pioneer.py
```

The script reads exact on-cluster CSVs and generates `paper_figures/exports/`.

The script reads real saved CSVs and writes six 300-dpi PNGs and six vector PDFs. The files in this GitHub folder are **SVG companions**, not byte-identical copies of the local package exports.

## Captions

See [FIGURE_CAPTIONS.md](FIGURE_CAPTIONS.md) for paper-ready captions and limitations.

## Repository organization

- `paper_figures/`: shareable figure assets and caption documentation.
- `final_results/`: existing summary tables and result figures.
- `experiment_a_outputs/`, `experiment_b_outputs/`: local Pioneer experiment artifacts; model checkpoints and slide-level feature files should not be committed without an explicit need and review.
- `create_paper_figures.py` (project root, after adding from the package): regenerate figures from local results.
