# Stability-Aware Candidate Selection for Model Fusion in Digital Pathology

**CSDS/EBME 463 — Project 2: Implementation of Recent AI Tools**

**Authors:** Aydin Gurudutt; [second group member — confirm]

**Date:** December 1, 2026

**Contribution statement (confirm before submission):** Aydin Gurudutt: stability-aware experiment and evaluation work (verify specific contributions). Second member: [name and technically specific contributions to be supplied]. These placeholders are not claims of actual contribution.

> **Submission-stage manuscript draft:** The companion PDF and PowerPoint were generated in ChatGPT and supplied there. This text is a version-controlled draft, **not a declaration that the originally proposed non-fused baseline was evaluated**. The paper must remain within six body pages excluding cover, references, and appendices.

## Abstract

Multiple instance learning (MIL) allows histopathology whole-slide classification without exhaustive patch annotations, but trained models can exhibit varying discriminative performance across random seeds. We investigated whether stability-aware candidate selection could improve multi-fidelity model fusion. Five candidate CLAM or DSMIL models were generated per seed on CAMELYON16. Their early validation ROC-AUC histories were ranked by a weighted mean minus a standard-deviation penalty, with the top three candidates merged using a TIES-style procedure and trained for two additional effective epochs. Across three seeds, CLAM baseline/stability-aware mean held-out test AUC was 0.7279/0.7103 with sample SD 0.0566/0.0259; DSMIL mean was 0.6205/0.6219 with sample SD 0.0570/0.0571. The minimum test AUC was unchanged under both selection strategies for both architectures. Archived SLURM records enabled comparison of candidate and continuation training time but not full end-to-end computational cost. These findings do not establish a general advantage for stability-based ranking. A previously proposed matched standalone non-fused comparison remains unmeasured.

## 1. Introduction

Whole-slide images (WSIs) contain massive numbers of histologic pixels. MIL treats a slide as a bag of patch representations and predicts a slide-level classification label rather than requiring a label for each patch. CLAM incorporates attention and clustering-based instance-level supervision [1]; DSMIL combines instance-level and bag-level evidence [2]. These architectures can be sensitive to stochastic training choices, complicating comparisons across model runs.

Mammadov et al. [3] proposed multi-fidelity model fusion, in which multiple candidates are cheaply trained, selected using validation results, and merged before continuation training. A potential weakness is that high early validation performance may be transient. We therefore incorporated variation across early checkpoints into the candidate-selection score.

**Hypothesis:** A stability penalty on early checkpoint AUC would decrease across-seed variation in the final held-out test AUC without substantially decreasing average discriminative performance.

## 2. Materials and methods

### Dataset and preprocessing

The clinical task was binary breast cancer lymph-node metastasis classification on CAMELYON16 [4]. The inherited pipeline used pre-extracted 512-dimensional patch features in slide-level CSV bags. The data were partitioned into 215 training, 54 validation and 128 held-out test slides. Training/validation/test slide filenames were checked for overlap. A separate patient-level independence audit and independent reproduction of the feature encoder were not available.

### Candidate training and model selection

Five candidate models were generated for each architecture at seeds 1, 2 and 3, and monitored for five early checkpoints. Learning rate was 0.0002 and weight decay was 0.005. CLAM used equal weights across the five validation AUC measurements; DSMIL used only the last two (nonzero checkpoint weights). For candidate i, the selection score was:

**S_i = weighted_mean_validation_AUC_i − λ × weighted_SD_validation_AUC_i**

Baseline selection used λ=0. Stability-aware selection used λ=0.5 (CLAM) or λ=5.0 (DSMIL). The three highest-ranked candidates were merged by a TIES-style mechanism relative to shared initialization [5]; the merged model underwent two effective additional training epochs. The final checkpoint was selected by validation AUC.

### Experiments and quantitative analysis

Two architectures × three seeds × two fusion-selection strategies produced 12 held-out test evaluations, each on the same 128-slide test split. The outcomes were test ROC AUC mean, sample standard deviation across seeds, minimum test AUC, highest validation-selected AUC, and within-seed paired AUC differences.

Archived Pioneer SLURM accounting provides completed job elapsed times for candidate generation and post-merge continuation. These are partial runtime records; they exclude separately executed offline merging, scheduling, and final evaluation. Mean and sample SD were computed for three seeds per architecture/selection strategy.

**Scope:** The original proposal also called for a non-fused standard MIL comparison. A single-best-candidate matched continuation/evaluation script is available, but its six GPU runs have not yet produced results. Earlier one-run ABMIL values were not included because they are not matched to this protocol.

## 3. Results

### Performance summary

| Architecture | Method | Test AUC mean ± sample SD | Minimum test AUC | Highest validation AUC |
|---|---|---:|---:|---:|
| CLAM | Baseline fusion | 0.7279 ± 0.0566 | 0.6949 | 0.8267 |
| CLAM | Stability-aware fusion | 0.7103 ± 0.0259 | 0.6949 | 0.7884 |
| DSMIL | Baseline fusion | 0.6205 ± 0.0570 | 0.5618 | 0.6264 |
| DSMIL | Stability-aware fusion | 0.6219 ± 0.0571 | 0.5618 | 0.6264 |

Here “highest validation AUC” is the maximum among the three validation-selected continuation checkpoints, not a separate experiment.

The mean paired held-out test AUC changes were −0.0176 for CLAM and +0.0013 for DSMIL. CLAM seed 1 decreased by −0.0530, while seeds 2 and 3 were identical or nearly unchanged. DSMIL seed 1 increased by +0.0040, and seeds 2–3 were unchanged. These measurements are published in final_results/all_results.csv and the derived extended performance table.

The mean **partial** elapsed time, candidate generation plus continuation, was approximately 22:47 (CLAM baseline), 21:51 (CLAM stability), 22:49 (DSMIL baseline), and 22:55 (DSMIL stability). The candidate stage was shared; these are per-condition hypothetical sums of measured stages, not independent duplicate candidate jobs or complete end-to-end time. See final_results/PROVENANCE.md and the 18 individually identified job durations.

### Figures

Figure 1: Held-out AUC mean and sample SD across the three seeds; generated from final_results/all_results.csv. Figure 2: Paired AUC differences, stability-aware minus baseline, per seed. Figure 3 in the companion PDF compares partial runtime. Original two chart assets are available in final_results/.

## 4. Discussion

The original hypothesis is only partly supported descriptively. Although CLAM's observed test-AUC SD decreased by about 54%, its mean AUC also fell; its minimum test AUC did not improve. Most variance reduction reflects a decrease in its strongest baseline seed, not an improvement in the worst seed. DSMIL changes were small and its variability effectively unchanged.

The experimental novelty is a variance-penalized candidate ranking criterion layered on the prior fusion framework; neither CLAM nor DSMIL's underlying architecture was modified. Architecture-specific weighting and λ values limit claims about universal applicability. The relatively small n=3, one dataset, some identical candidate selections, missing full runtime breakdown, lack of patient-level split audit, and unmeasured standalone comparison all limit interpretability.

Future work should execute the non-fused single-model arm, expand random seeds and datasets, test λ on prespecified validation protocols, analyze actual candidate membership changes, and capture offline merge/total runtime.

## 5. Conclusion

Early validation stability can be included in MIL fusion candidate selection, but the present results do not demonstrate reliable improvement in held-out classification. The CLAM decrease in observed variation accompanies lower mean accuracy and unchanged worst-seed performance, while DSMIL changes are negligible. Larger matched experiments are needed before making broader reproducibility or deployment claims.

## References

1. Lu, M. Y., et al. (2021). Data-efficient and weakly supervised computational pathology on whole-slide images. *Nature Biomedical Engineering*, 5, 555–570. https://doi.org/10.1038/s41551-020-00682-w
2. Li, B., Li, Y., & Eliceiri, K. W. (2021). Dual-stream multiple instance learning network for whole slide image classification with self-supervised contrastive learning. *CVPR*, 14318–14328. https://doi.org/10.1109/CVPR46437.2021.01409
3. Mammadov, A., Le Folgoc, L., Hocquet, G., & Gori, P. (2025). Reducing Variability of Multiple Instance Learning Methods for Digital Pathology. *MICCAI 2025*, 506–516. https://doi.org/10.1007/978-3-032-04981-0_48
4. Ehteshami Bejnordi, B., et al. (2017). Diagnostic assessment of deep learning algorithms for detection of lymph node metastases in women with breast cancer. *JAMA*, 318(22), 2199–2210.
5. Yadav, P., Tam, D., Choshen, L., Raffel, C., & Bansal, M. (2023). TIES-Merging: Resolving interference when merging models. *NeurIPS*, 36. https://arxiv.org/abs/2306.01708

Attribution: https://github.com/mammadov7/mil_merging; original group project https://github.com/karthikraj2103/Final-project-CSDS463; current project https://github.com/swimstingray/Final-project-CSDS463.

## Appendix: dataset/code provenance and outstanding checks

Important scripts: train_candidates_clam.py, train_candidates_dsmil.py, train_repro_clam.py, train_repro_dsmil.py, train_fusion_clam.py, train_fusion_dsmil.py, evaluate_heldout.py, make_final_results.py, standalone_comparison.py. Patient-level split independence has not been audited and model checkpoint files are excluded from the GitHub repository. The standalone comparator, second-author contribution verification, code ZIP, and actual Canvas submission remain necessary before declaring Project 2 fully complete.
