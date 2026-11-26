#!/usr/bin/env python3
"""Prepare two controlled ABMIL weighted-fusion arms without using the test set.

Same top-three candidates in both arms, selected by mean early validation AUC.
New A: weights from softmax(mean AUC / temperature).
New B: weights from softmax((mean AUC - lambda*SD) / temperature).
Both use weighted sign-consensus merging, k=1 (no trimming), and shared init.
"""
import json
from pathlib import Path
import numpy as np
import torch

RUNS={1:"20261125152831551498",2:"20261125152841024449",3:"20261125154225498903"}
ROOT=Path("weights/abmil")
BASE=Path("weights/abmil_fusions")
OUT=Path("weights/abmil_weighted_fusions")
PENALTY=2.0
TEMPERATURE=0.02  # Fixed before comparing the two arms; exploratory setting.

def load(path):
    return torch.load(path,map_location="cpu",weights_only=True)

def softmax(x):
    z=np.asarray(x,dtype=np.float64)/TEMPERATURE
    w=np.exp(z-z.max())
    w/=w.sum()
    assert np.isfinite(w).all() and (w>0).all()
    return w

def merge(states,init,weights):
    """Weighted sign election and disjoint weighted mean of task vectors."""
    weights=np.asarray(weights,dtype=float)
    assert len(states)==len(weights) and np.isclose(weights.sum(),1)
    result={}
    for key,start in init.items():
        if any(key not in s or s[key].shape!=start.shape for s in states):
            raise ValueError(f"Incompatible state: {key}")
        if not start.is_floating_point():
            result[key]=start.clone()
            continue
        delta=torch.stack([s[key]-start for s in states],dim=0)
        if not torch.isfinite(delta).all():
            raise ValueError(f"Nonfinite candidate: {key}")
        w=torch.as_tensor(weights,dtype=delta.dtype).reshape(
            (len(states),)+(1,)*start.ndim)
        vote=torch.sign((delta*w).sum(dim=0))
        agree=(torch.sign(delta)==vote)&(vote!=0)
        numerator=(delta*w*agree).sum(dim=0)
        denominator=(w*agree).sum(dim=0)
        update=torch.where(denominator>0,numerator/denominator.clamp_min(1e-30),
                           torch.zeros_like(numerator))
        result[key]=start+update
        if not torch.isfinite(result[key]).all():
            raise ValueError(f"Nonfinite merge: {key}")
    return result

def distance(a,b):
    assert set(a)==set(b)
    return max(float((a[k]-b[k]).abs().max()) for k in a
               if a[k].is_floating_point() and a[k].numel())

def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing overwrite: {OUT}")
    prepared=[]
    for seed,run_id in RUNS.items():
        folder=ROOT/run_id
        with np.load(folder/"candidate_histories.npz",allow_pickle=False) as f:
            aucs=np.asarray(f["early_aucs"],dtype=float)
            assert int(f["seed"])==seed
        if aucs.shape!=(5,5) or not np.isfinite(aucs).all():
            raise ValueError(f"Bad history for seed {seed}")
        mean=aucs.mean(axis=1)
        std=aucs.std(axis=1,ddof=0)
        top=np.argsort(mean)[::-1][:3].tolist()
        old=BASE/f"seed_{seed}"
        meta=json.loads((old/"selection_metadata.json").read_text())
        if meta["candidate_run"]!=run_id or set(top)!=set(
                meta["selections"]["baseline"]["selected_candidates"]):
            raise ValueError(f"Candidate selection mismatch for seed {seed}")
        initial=load(folder/"initial_weights.pth")
        selected=[load(folder/"candidates"/f"candidate_{i}.pth") for i in top]
        equal=merge(selected,initial,[1/3]*3)
        original=load(old/"baseline"/"merged_initial.pth")
        baseline_error=distance(equal,original)
        if baseline_error>1e-6:
            raise ValueError(f"Equal merge differs from baseline for seed {seed}: "
                             f"{baseline_error}")
        auc_weights=softmax(mean[top])
        stability_weights=softmax(mean[top]-PENALTY*std[top])
        auc_merged=merge(selected,initial,auc_weights)
        stability_merged=merge(selected,initial,stability_weights)
        effect=distance(auc_merged,stability_merged)
        if effect<=0:
            raise ValueError(f"Weighted arms identical for seed {seed}")
        details={
            "architecture":"ABMIL","seed":seed,"candidate_run":run_id,
            "selected_candidates":top,"mean_validation_auc":mean[top].tolist(),
            "std_validation_auc":std[top].tolist(),
            "stability_penalty":PENALTY,"temperature":TEMPERATURE,
            "trim_fraction_kept":1.0,
            "auc_weights":auc_weights.tolist(),
            "stability_weights":stability_weights.tolist(),
            "max_difference_between_arms":effect,
            "max_difference_auc_vs_equal":distance(auc_merged,equal),
            "max_difference_stability_vs_equal":distance(stability_merged,equal),
            "equal_weight_reference_error":baseline_error,
            "test_data_accessed":False
        }
        print(f"Seed {seed}: top={top} AUC weights="
              f"{np.round(auc_weights,4).tolist()} stability weights="
              f"{np.round(stability_weights,4).tolist()} "
              f"max_difference={effect:.6g} equal_baseline_error="
              f"{baseline_error:.2e}",flush=True)
        prepared.append((seed,details,auc_merged,stability_merged))
    for seed,details,auc,stable in prepared:
        dest=OUT/f"seed_{seed}"
        for method,state in (("auc_weighted",auc),("stability_weighted",stable)):
            target=dest/method
            target.mkdir(parents=True,exist_ok=False)
            torch.save(state,target/"merged_initial.pth")
        (dest/"merge_metadata.json").write_text(json.dumps(details,indent=2)+"\n")
    print("PASS: three seeds; original equal-weight baseline reproduced; distinct weighted merges")
    print("SAVED:",OUT)
    print("No held-out test data used.")

if __name__=="__main__":
    main()
