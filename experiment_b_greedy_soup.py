#!/usr/bin/env python3
"""Validation-only greedy equal-weight model soups (no test data)."""
import argparse, csv, json, math, random
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from experiment_a_ensemble import load_split, RUNS
import evaluate_heldout as evaluation
import train_repro_clam as clam
import train_repro_dsmil as dsmil
from train_fusion_dsmil import build_model

ABMIL = {1:"20261125152831551498", 2:"20261125152841024449", 3:"20261125154225498903"}
OUT = Path("experiment_b_outputs")
EPS = 1e-8

def merge(states):
    if not states: raise ValueError("Empty model soup")
    keys = list(states[0])
    if any(set(s) != set(keys) for s in states): raise ValueError("Different state keys")
    result = {}
    for k in keys:
        ts = [s[k] for s in states]
        if any(t.shape != ts[0].shape or t.dtype != ts[0].dtype for t in ts):
            raise ValueError(f"Incompatible tensor {k}")
        if ts[0].is_floating_point():
            if any(not torch.isfinite(t).all() for t in ts):
                raise ValueError(f"Nonfinite tensor {k}")
            result[k] = torch.stack([t.double() for t in ts]).mean(0).to(ts[0].dtype)
        else:
            if any(not torch.equal(t,ts[0]) for t in ts[1:]):
                raise ValueError(f"Nonfloating tensor mismatch {k}")
            result[k] = ts[0].clone()
    return result

def score(arch, state, bags, labels):
    args = evaluation.make_args(arch)
    model = build_model(args) if arch == "dsmil" else clam.get_model(args)
    model.load_state_dict(state, strict=True)
    model.cuda().eval()
    preds = []
    with torch.inference_mode():
        for x in bags:
            if arch == "dsmil":
                ins, bag, _, _ = model(x)
                max_ins, _ = torch.max(ins, dim=0)
                p = float(np.asarray(dsmil.activation(args, bag, max_ins)).reshape(-1)[0])
            else:
                bag, _ = model(x)
                p = float(torch.sigmoid(bag).reshape(-1)[0].item())
            if not math.isfinite(p) or not 0 <= p <= 1:
                raise ValueError("Invalid probability")
            preds.append(p)
    del model
    return float(roc_auc_score(labels,preds))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--architecture", choices=("clam","abmil","dsmil"), required=True)
    ap.add_argument("--seed", type=int, choices=(1,2,3), required=True)
    a = ap.parse_args()
    arch, seed = a.architecture, a.seed
    out = OUT / arch / f"seed_{seed}"
    if out.exists(): raise FileExistsError(f"Refusing overwrite: {out}")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    source = Path("weights/abmil") / ABMIL[seed] if arch=="abmil" else Path("weights") / arch / RUNS[arch][seed]
    with np.load(source/"candidate_histories.npz", allow_pickle=False) as f:
        hist=np.asarray(f["early_aucs"], dtype=float)
        if int(f["seed"])!=seed or hist.shape!=(5,5) or not np.isfinite(hist).all():
            raise ValueError("Candidate provenance/history mismatch")
    paths=[source/"candidates"/f"candidate_{i}.pth" for i in range(5)]
    for p in paths:
        if not p.is_file() or p.stat().st_size==0: raise FileNotFoundError(p)
    states=[torch.load(p,map_location="cpu",weights_only=True) for p in paths]
    merge(states)  # Check state compatibility before inference.
    val=load_split().reset_index(drop=True)
    labels=val["label"].astype(int).to_numpy()
    random.seed(12345); np.random.seed(12345)
    torch.manual_seed(12345); torch.cuda.manual_seed_all(12345)
    args=evaluation.make_args(arch)
    loader=dsmil if arch=="dsmil" else clam
    bags=[]
    for _,row in val.iterrows():
        label, data=loader.get_bag_feats(row,args,random_state=1)
        if int(np.asarray(label).reshape(-1)[0])!=int(row["label"]):
            raise ValueError(f"Label mismatch: {row['path']}")
        data=np.asarray(data,dtype=np.float32)
        if data.ndim!=2 or data.shape[1]!=512 or not np.isfinite(data).all():
            raise ValueError(f"Invalid features: {row['path']}")
        bags.append(torch.from_numpy(data.copy()).cuda())
    print(f"PREFLIGHT PASS arch={arch} seed={seed} slides={len(val)} candidates=5",flush=True)
    singles=[]
    for i,state in enumerate(states):
        s=score(arch,state,bags,labels)
        singles.append(s)
        print(f"SINGLE candidate={i} AUC={s:.8f}",flush=True)
    start=min(range(5),key=lambda i:(-singles[i],i))
    selected=[start]; current=singles[start]; trace=[(0,"initial",start,selected.copy(),current,0)]
    remaining=set(range(5))-{start}
    while remaining:
        trials=[]
        for i in sorted(remaining):
            trial=score(arch,merge([states[j] for j in selected+[i]]),bags,labels)
            trials.append((trial,i))
            print(f"TRY chosen={selected} add={i} AUC={trial:.8f}",flush=True)
        best_auc=max(t[0] for t in trials)
        best_i=min(t[1] for t in trials if abs(t[0]-best_auc)<=EPS)
        if best_auc<=current+EPS:
            print("STOP no strict improvement",flush=True)
            break
        previous=current; current=best_auc
        selected.append(best_i); remaining.remove(best_i)
        trace.append((len(selected)-1,"accepted",best_i,selected.copy(),current,current-previous))
        print(f"ACCEPT chosen={selected} AUC={current:.8f}",flush=True)
    final=merge([states[j] for j in selected])
    if abs(score(arch,final,bags,labels)-current)>EPS:
        raise ValueError("Final AUC reproduction failed")
    out.mkdir(parents=True,exist_ok=False)
    torch.save(final,out/"greedy_soup.pth")
    with (out/"selection_trace.csv").open("x",newline="") as f:
        w=csv.writer(f); w.writerow(["step","action","candidate","selected","validation_auc","delta"])
        for step,act,candidate,chosen,auc,delta in trace:
            w.writerow([step,act,candidate,json.dumps(chosen),auc,delta])
    (out/"manifest.json").write_text(json.dumps({
        "experiment":"B: greedy arithmetic checkpoint soup",
        "architecture":arch,"seed":seed,"candidate_paths":[str(p) for p in paths],
        "validation_slides":len(val),"single_candidate_auc":singles,
        "best_single_auc":singles[start],"selected_indices":selected,
        "greedy_validation_auc":current,"delta_vs_best_single":current-singles[start],
        "selected_checkpoint":str(out/"greedy_soup.pth"),
        "heldout_test_accessed":False,"post_merge_training":False,
        "selection_auc_optimistically_biased":True
    },indent=2)+"\n")
    print(f"PASS arch={arch} seed={seed} selected={selected} AUC={current:.8f}",flush=True)

if __name__=="__main__": main()
