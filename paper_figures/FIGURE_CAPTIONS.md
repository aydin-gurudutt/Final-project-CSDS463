# Figure captions

**Figure 1 — Experimental workflow.** Overview of CAMELYON16 MIL methods with 215 training, 54 validation and 128 held-out test slides. CLAM, DSMIL and ABMIL were used in separate standalone, TIES-style fusion, and exploratory ensemble/soup experiments. Experiments A and B were evaluated only on validation. Not all architectures used identical continuation protocols.

**Figure 2 — Held-out model comparisons.** Slide-level test ROC AUC for CLAM and DSMIL standalone checkpoints, baseline TIES-style merging, and stability-aware merging. Bars represent the mean over three seeds; whiskers represent sample SD, not confidence intervals. Observed differences do not establish statistical significance.

**Figure 3 — Observed stability–performance trade-off.** Mean held-out ROC AUC versus sample SD across three seeds. CLAM stability-aware selection reduced observed variability (0.0259 compared with 0.0566) at a lower mean AUC (0.7103 compared with 0.7279). DSMIL changes were small. Three seeds give imprecise variability estimates.

**Figure 4 — Paired seed-level differences.** Test ROC AUC change for stability-aware minus baseline candidate selection at matched architecture/seed. CLAM seed 1 showed the largest decrease (approximately -0.0530); several comparisons were unchanged. Identical outcomes can result from identical selections.

**Figure 5 — Experiment A: heterogeneous probability ensembles.** Mean validation AUC across three seeds for CLAM alone, CLAM+ABMIL, and CLAM+ABMIL+DSMIL, separately for standalone and equal-merge checkpoint families. Fixed, equally weighted probability averaging provided no consistent benefit. All measurements are validation-only.

**Figure 6 — Experiment B: greedy checkpoint soup.** Validation AUC for the strongest original candidate versus greedy equal-weight parameter averaging in each architecture/seed. Of nine runs, only CLAM seed 1 accepted additional checkpoints and gained approximately 0.00284 AUC. The same 54-slide validation set selected the soup and measured the apparent gain.

## Reporting constraints

The source dataset and checkpoint weights are stored on Pioneer, not in this folder. No patient-level independence has been verified. Weighted TIES variants require performance extraction before they are plotted. Keep held-out and validation-only comparisons in separate figures and text.
