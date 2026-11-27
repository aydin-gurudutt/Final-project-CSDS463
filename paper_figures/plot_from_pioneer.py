#!/usr/bin/env python3
"""Regenerate six publication figures using exact Pioneer experiment CSV files.

Run from repository root:
  python paper_figures/plot_from_pioneer.py

Writes high-resolution PNG and vector PDF to paper_figures/exports.
Does not train or infer on any dataset. Does not overwrite existing files.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper_figures" / "exports"
FILES = {
    "summary": ROOT / "final_results" / "summary.csv",
    "three": ROOT / "final_results" / "three_way_summary.csv",
    "all": ROOT / "final_results" / "all_results.csv",
    "a": ROOT / "experiment_a_outputs" / "validation_auc_results.csv",
    "b": ROOT / "final_results" / "experiment_b_summary.csv",
}
COLORS = ("#2777a3", "#218a81", "#cd7444")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})

def save(fig, number, slug):
    for suffix in ("png", "pdf"):
        target = OUT / ("figure_%02d_%s.%s" % (number, slug, suffix))
        if target.exists():
            raise FileExistsError("Refusing to overwrite " + str(target))
        fig.savefig(target, dpi=300, bbox_inches="tight", facecolor="white")
        print("SAVED", target, flush=True)
    plt.close(fig)

def validate(d):
    expected = {"summary":(4,{"architecture","method","test_mean","test_sd"}),
                "three":(6,{"architecture","method","test_mean","test_sample_sd"}),
                "all":(12,{"architecture","seed","method","test_auc"}),
                "a":(18,{"family","seed","method","validation_auc"}),
                "b":(9,{"architecture","seed","best_single_validation_auc","greedy_validation_auc"})}
    for name,(n,columns) in expected.items():
        if len(d[name]) != n or not columns.issubset(d[name].columns):
            raise ValueError("Unexpected schema or count for " + str(FILES[name]))
    for name in ("a","b"):
        df=d[name]
        if set(df["seed"].astype(int)) != {1,2,3}:
            raise ValueError("Wrong seeds in " + name)

def main():
    d={}
    for name,path in FILES.items():
        if not path.is_file():
            raise FileNotFoundError("Missing source table: " + str(path))
        d[name]=pd.read_csv(path)
    validate(d)
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError("Output folder is not empty: " + str(OUT))
    OUT.mkdir(parents=True,exist_ok=True)

    # Figure 1: Methods schematic.
    fig,ax=plt.subplots(figsize=(11,6));ax.set(xlim=(0,12),ylim=(0,8));ax.axis("off")
    def node(x,y,w,h,heading,detail):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.09",
                                   edgecolor="#9fb2c4",facecolor="#edf3f7"))
        ax.text(x+w/2,y+h*.68,heading,weight="bold",ha="center",fontsize=10)
        ax.text(x+w/2,y+h*.28,detail,ha="center",fontsize=9)
    def arrow(x,y,xx,yy):
        ax.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle="-|>",
                                    mutation_scale=14,color="#60707f"))
    node(3.5,6.5,5,1,"CAMELYON16 feature bags","215 train / 54 validation / 128 held-out test")
    node(.2,4.3,3.4,1.1,"Candidate architectures","CLAM / DSMIL / ABMIL")
    node(4.3,4.3,3.4,1.1,"Model selection & fusion","Standalone / TIES / stability / weighted")
    node(8.4,4.3,3.4,1.1,"Exploratory experiments","A: ensembles / B: greedy soup")
    node(.6,1.5,4.8,1.15,"Validation-only","Experiments A and B")
    node(6.6,1.5,4.8,1.15,"Existing held-out test","Standalone / baseline / stability")
    for args in [(6,6.5,1.9,5.4),(6,6.5,6,5.4),(6,6.5,10.1,5.4),
                 (1.9,4.3,3,2.65),(6,4.3,3.3,2.65),
                 (6,4.3,8.6,2.65),(10.1,4.3,9,2.65)]:
        arrow(*args)
    ax.set_title("Experimental workflow and data boundaries",weight="bold")
    ax.text(6,.4,"Not all method arms use identical training/continuation protocols.",
            ha="center",fontsize=9)
    save(fig,1,"workflow")

    # Figure 2: Held-out mean and sample SD.
    three=d["three"]; fig,ax=plt.subplots(figsize=(9.2,5))
    archs=["CLAM","DSMIL"]; methods=["Standalone","Baseline","Stability-aware"]
    x=np.arange(2);width=.23
    for j,method in enumerate(methods):
        sub=three[three.method==method].set_index("architecture").loc[archs]
        ax.bar(x+(j-1)*width,sub.test_mean,width,yerr=sub.test_sample_sd,
               capsize=4,color=COLORS[j],label=method)
    ax.set(xticks=x,xticklabels=archs,ylim=(0,.9),ylabel="Held-out slide-level ROC AUC",
           title="Held-out performance by architecture and method")
    ax.legend(frameon=False);ax.grid(axis="y",alpha=.2)
    fig.text(.5,.01,"Mean ± sample SD over 3 seeds. Not confidence intervals.",
             ha="center",fontsize=9)
    save(fig,2,"heldout_methods")

    # Figure 3: Mean vs between-seed SD.
    fig,ax=plt.subplots(figsize=(8.4,5))
    for i,arch in enumerate(archs):
        sub=d["summary"][d["summary"].architecture==arch].set_index("method")
        points=sub.loc[["Baseline","Stability-aware"],["test_sd","test_mean"]]
        ax.plot(points.test_sd,points.test_mean,"-o",color=COLORS[i],label=arch)
        for name,row in points.iterrows():
            ax.annotate(arch+" "+name,(row.test_sd,row.test_mean),
                        xytext=(5,5 if name=="Baseline" else -13),
                        textcoords="offset points",fontsize=8)
    ax.set(xlabel="Across-seed sample SD of held-out AUC",
           ylabel="Mean held-out AUC",title="Stability versus predictive performance")
    ax.grid(alpha=.2)
    save(fig,3,"stability_tradeoff")

    # Figure 4: Paired changes from the same seeds.
    fig,axs=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for i,arch in enumerate(archs):
        sub=d["all"][d["all"].architecture==arch].pivot(
            index="seed",columns="method",values="test_auc").sort_index()
        delta=sub["Stability-aware"]-sub["Baseline"]
        axs[i].bar(sub.index.astype(str),delta.values,color=COLORS[i])
        axs[i].axhline(0,color="#555",linewidth=1)
        axs[i].set(xlabel="Seed",title=arch+" (mean delta %.4f)"%delta.mean())
        axs[i].grid(axis="y",alpha=.2)
    axs[0].set_ylabel("Held-out AUC: stability-aware minus baseline")
    fig.suptitle("Paired per-seed differences")
    save(fig,4,"seed_paired_test_changes")

    # Figure 5: Ensemble validation AUC.
    fig,axs=plt.subplots(1,2,figsize=(11,4.8),sharey=True)
    methods=["CLAM","CLAM+ABMIL","CLAM+ABMIL+DSMIL"]
    for i,family in enumerate(["standalone","baseline"]):
        sub=d["a"][d["a"].family==family]
        ag=sub.groupby("method").validation_auc.agg(["mean","std"]).loc[methods]
        axs[i].bar(np.arange(3),ag["mean"],yerr=ag["std"],capsize=4,
                   color=COLORS)
        axs[i].set(xticks=np.arange(3),xticklabels=["CLAM","CLAM + ABMIL","All 3"],
                   ylim=(0,.9),title=family.capitalize()+" checkpoints")
        axs[i].grid(axis="y",alpha=.2)
    axs[0].set_ylabel("Validation ROC AUC")
    fig.suptitle("Experiment A: equal-probability architecture ensembles")
    fig.text(.5,.01,"Mean ± sample SD over 3 seeds; validation only.",ha="center")
    save(fig,5,"experiment_a_ensembles")

    # Figure 6: Single candidate versus selected greedy soup per run.
    fig,axs=plt.subplots(1,3,figsize=(12.5,4.6),sharey=True)
    for i,arch in enumerate(["CLAM","ABMIL","DSMIL"]):
        sub=d["b"][d["b"].architecture==arch].sort_values("seed")
        xx=np.arange(3);width=.36
        axs[i].bar(xx-width/2,sub.best_single_validation_auc,width,
                   label="Best single",color=COLORS[0])
        axs[i].bar(xx+width/2,sub.greedy_validation_auc,width,
                   label="Greedy",color=COLORS[1])
        axs[i].set(xticks=xx,xticklabels=["Seed 1","Seed 2","Seed 3"],
                   ylim=(0,.9),title=arch)
        axs[i].grid(axis="y",alpha=.2)
    axs[0].set_ylabel("Validation ROC AUC")
    axs[1].legend(frameon=False,loc="lower center")
    fig.suptitle("Experiment B: greedy checkpoint averaging")
    fig.text(.5,.01,"Validation selection and evaluation use the same 54 slides.",
             ha="center",fontsize=9)
    save(fig,6,"experiment_b_greedy_soups")

if __name__=="__main__":
    main()
