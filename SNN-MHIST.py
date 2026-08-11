import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import defaultdict
from sklearn.decomposition import PCA

BACH_ROOT  = "/kaggle/input/datasets/preanto/bach-files"
MHIST_ROOT = "/kaggle/input/datasets/preanto/snn-mhist-output"
MHIST_OUT  = "/kaggle/working/mhist_paper_artifacts"
CROSS_OUT  = "/kaggle/working/cross_dataset_artifacts"
os.makedirs(MHIST_OUT, exist_ok=True)
os.makedirs(CROSS_OUT, exist_ok=True)

# ─── Load everything ──────────────────────────────────────────────────
def load_run(root, exp, seed, T, dataset):
    prefix = "bach" if dataset == "BACH" else "mhist"
    d = os.path.join(root, f"{prefix}_{exp}_T{T}_seed{seed}")
    with open(os.path.join(d, "sidecar.json")) as f:
        cfg = json.load(f)
    npz = np.load(os.path.join(d, "analysis.npz"), allow_pickle=True)
    bh_path = os.path.join(d, "beta_history.csv")
    beta_hist = pd.read_csv(bh_path) if os.path.exists(bh_path) else None
    return {"cfg": cfg, "npz": npz, "beta_history": beta_hist,
            "dataset": dataset,
            "label": f"{dataset}_{exp}_s{seed}_T{T}"}

BACH_RUNS = [
    ("baseline_learn",  42,   16), ("baseline_learn",  123,  16), ("baseline_learn",  2024, 16),
    ("matches",         42,   16), ("matches",         123,  16), ("matches",         2024, 16),
    ("baseline_frozen", 42,   16),
    ("baseline_learn",  42,    4),
]
MHIST_RUNS = [
    ("baseline_learn", 42, 16), ("baseline_learn", 123, 16), ("baseline_learn", 2024, 16),
    ("matches",        42, 16), ("matches",        123, 16), ("matches",        2024, 16),
]

bach = [load_run(BACH_ROOT, *r, "BACH") for r in BACH_RUNS]
mhist = [load_run(MHIST_ROOT, *r, "MHIST") for r in MHIST_RUNS]

COLORS  = {"baseline_learn": "tab:blue", "matches": "tab:orange", "baseline_frozen": "tab:green"}
MARKERS = {42: "o", 123: "s", 2024: "^"}
print(f"Loaded BACH: {len(bach)} runs;  MHIST: {len(mhist)} runs")

# ══════════════════════════════════════════════════════════════════════
# MHIST STANDALONE FIGURES (parallel to BACH fig4-13)
# ══════════════════════════════════════════════════════════════════════

def label_style(r):
    cfg = r["cfg"]
    if cfg["T"] == 4:
        return dict(color="tab:red", marker="D", linestyle="--", label=f"baseline_learn s42 T=4")
    return dict(color=COLORS.get(cfg["experiment"], "gray"),
                marker=MARKERS.get(cfg["seed"], "x"), linestyle="-",
                label=f"{cfg['experiment']} s{cfg['seed']}")

# ── MHIST Fig 4: Temporal collapse (consec cosine) ────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
for r in mhist:
    T = r["cfg"]["T"]
    consec = r["npz"]["consec"].item()
    # MHIST uses HP/SSA labels
    ch = consec.get("HP", consec.get("benign", []))
    cs = consec.get("SSA", consec.get("malignant", []))
    avg = [(h+s)/2 for h,s in zip(ch, cs)]
    st = label_style(r)
    ax.plot(range(1, T), avg, **st, alpha=0.75, linewidth=1.5, markersize=6)
ax.axhline(0.998, color="gray", linestyle=":", alpha=0.5, label="saturation ≈ 0.998")
ax.set_xlabel("timestep transition t → t+1"); ax.set_ylabel("consecutive-timestep feature cosine")
ax.set_title("MHIST Fig 4 — Temporal collapse replicates on MHIST across 6 runs")
ax.set_ylim(0.9, 1.001); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right", ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(MHIST_OUT, "mhist_fig4_temporal_collapse.png"), dpi=200, bbox_inches="tight")
plt.savefig(os.path.join(MHIST_OUT, "mhist_fig4_temporal_collapse.pdf"), bbox_inches="tight")
plt.show()

# ── MHIST Fig 5: Probe AUC across timesteps ───────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
for r in mhist:
    probe = r["npz"]["probe"].item()
    ts = sorted(probe.keys())
    means = [probe[t][0] for t in ts]; stds = [probe[t][1] for t in ts]
    ax.errorbar(ts, means, yerr=stds, **label_style(r), alpha=0.75, capsize=3, markersize=6)
ax.set_xlabel("timestep"); ax.set_ylabel("5-fold CV probe AUC (test features)")
ax.set_title("MHIST Fig 5 — Feature-quality probe AUC (tighter seed variance than BACH)")
ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right", ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(MHIST_OUT, "mhist_fig5_probe_auc.png"), dpi=200, bbox_inches="tight")
plt.savefig(os.path.join(MHIST_OUT, "mhist_fig5_probe_auc.pdf"), bbox_inches="tight")
plt.show()

# ── MHIST Fig 6: Test AUC bar chart ───────────────────────────────────
groups = defaultdict(list)
for r in mhist:
    aucs = r["npz"]["test_aucs"].item()
    groups[r["cfg"]["experiment"]].append(aucs[r["cfg"]["T"]])
labels = list(groups.keys())
means = [np.mean(v) for v in groups.values()]
stds  = [np.std(v)  for v in groups.values()]
ns    = [len(v)     for v in groups.values()]
fig, ax = plt.subplots(figsize=(7, 5))
xs = np.arange(len(labels))
ax.bar(xs, means, yerr=stds, capsize=6, alpha=0.85,
       color=[COLORS[l] for l in labels])
for i, (m, s, n) in enumerate(zip(means, stds, ns)):
    ax.text(xs[i], m + s + 0.005, f"n={n}\n{m:.3f}±{s:.3f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(xs); ax.set_xticklabels(labels)
ax.set_ylabel("Test AUC (t=16)"); ax.set_title("MHIST Fig 6 — Test AUC by configuration")
ax.set_ylim(0.75, 0.87); ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(MHIST_OUT, "mhist_fig6_test_auc_bar.png"), dpi=200, bbox_inches="tight")
plt.savefig(os.path.join(MHIST_OUT, "mhist_fig6_test_auc_bar.pdf"), bbox_inches="tight")
plt.show()

# ── MHIST Fig 7: β evolution (baseline_learn, 3 seeds avg) ────────────
bl = [r for r in mhist if r["cfg"]["experiment"] == "baseline_learn" and r["beta_history"] is not None]
if bl:
    layer_cols = [c for c in bl[0]["beta_history"].columns if c.endswith(".beta")]
    fig, ax = plt.subplots(figsize=(11, 6))
    for lc in layer_cols:
        dfs = []
        for r in bl:
            bh = r["beta_history"][["epoch", lc]].copy(); bh.columns = ["epoch","beta"]; dfs.append(bh)
        avg = pd.concat(dfs).groupby("epoch")["beta"].mean().reset_index()
        ax.plot(avg["epoch"], avg["beta"], marker="o", markersize=3, linewidth=1.4,
                label=lc.replace(".beta",""))
    ax.axhline(0.95, color="gray", linestyle=":", alpha=0.5, label="init β=0.95")
    ax.set_xlabel("epoch"); ax.set_ylabel("mean β (learn_beta=True, avg over 3 seeds)")
    ax.set_title("MHIST Fig 7 — β evolution during training")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(MHIST_OUT, "mhist_fig7_beta_evolution.png"), dpi=200, bbox_inches="tight")
    plt.savefig(os.path.join(MHIST_OUT, "mhist_fig7_beta_evolution.pdf"), bbox_inches="tight")
    plt.show()

# ── MHIST Fig 10: Pairwise cosine (one heatmap per experiment) ────────
seen = set(); heatmap_runs = []
for r in mhist:
    key = r["cfg"]["experiment"]
    if key not in seen:
        seen.add(key); heatmap_runs.append(r)
fig, axes = plt.subplots(1, len(heatmap_runs), figsize=(5*len(heatmap_runs), 4.5))
if len(heatmap_runs) == 1: axes = [axes]
for ax_, r in zip(axes, heatmap_runs):
    ph = r["npz"]["pairwise_cos_HP"]; ps = r["npz"]["pairwise_cos_SSA"]
    avg = (ph + ps) / 2
    im = ax_.imshow(avg, cmap="viridis", vmin=0.85, vmax=1.0, origin="lower")
    ax_.set_title(f"{r['cfg']['experiment']} s{r['cfg']['seed']}")
    ax_.set_xlabel("t_i"); ax_.set_ylabel("t_j"); plt.colorbar(im, ax=ax_, fraction=0.046, pad=0.04)
plt.suptitle("MHIST Fig 10 — Pairwise feature cosine (T=16)", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(MHIST_OUT, "mhist_fig10_pairwise_cosine.png"), dpi=200, bbox_inches="tight")
plt.savefig(os.path.join(MHIST_OUT, "mhist_fig10_pairwise_cosine.pdf"), bbox_inches="tight")
plt.show()

# ── MHIST Fig 12: Accuracy vs SynOps ──────────────────────────────────
ann_macs = sum(mhist[0]["npz"]["macs"].item().values())
fig, ax = plt.subplots(figsize=(11, 6.5))
ax.axvline(ann_macs, color="black", linestyle=":", alpha=0.8, linewidth=2)
ax.axvspan(ann_macs, 1e10, alpha=0.05, color="red")
ax.text(ann_macs, 0.855, f"ANN forward\n= {ann_macs/1e8:.2f}×10⁸ ops",
        ha="center", va="top", fontsize=9, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black"))
for r in mhist:
    T = r["cfg"]["T"]
    total_cum = r["npz"]["synops_total_cum"]
    aucs = r["npz"]["test_aucs"].item()
    ts_plot = [t for t in [1,4,8,12,16] if t <= T]
    xs = [total_cum[t-1] for t in ts_plot]; ys = [aucs[t] for t in ts_plot]
    ax.plot(xs, ys, **label_style(r), alpha=0.75, markersize=7, linewidth=1.4)
ax.set_xscale("log")
tick_locs = [1e8, 2e8, 3e8, 5e8, 7e8, 1e9, 2e9, 3e9, 5e9]
ax.set_xticks(tick_locs)
def fmt(v, pos): return f"{v/1e8:.0f}×10⁸" if v < 1e9 else f"{v/1e9:.0f}×10⁹"
ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt))
ax.set_xlim(1e8, 5e9); ax.set_ylim(0.58, 0.87)
ax.set_yticks(np.arange(0.60, 0.86, 0.02))
ax.set_xlabel("Cumulative SynOps (log scale)"); ax.set_ylabel("Test AUC at timestep t")
ax.set_title("MHIST Fig 12 — Same waste-zone pattern as BACH:\n"
             "AUC saturates ~t=4 (~1.5× ANN); compute grows to ~6.5× ANN with marginal AUC gain")
ax.grid(True, which="major", alpha=0.4); ax.grid(True, which="minor", alpha=0.15, linestyle=":")
ax.legend(fontsize=8, loc="lower right", ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(MHIST_OUT, "mhist_fig12_accuracy_vs_synops.png"), dpi=200, bbox_inches="tight")
plt.savefig(os.path.join(MHIST_OUT, "mhist_fig12_accuracy_vs_synops.pdf"), bbox_inches="tight")
plt.show()

# ══════════════════════════════════════════════════════════════════════
# CROSS-DATASET COMPARISON FIGURES (paper §4.5)
# ══════════════════════════════════════════════════════════════════════

# ── Cross Fig A: Temporal collapse — BACH vs MHIST side-by-side ───────
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ax_, runs_set, title in [(axes[0], bach, "BACH (n=400)"),
                             (axes[1], mhist, "MHIST (n=3152)")]:
    for r in runs_set:
        T = r["cfg"]["T"]; consec = r["npz"]["consec"].item()
        # Handle both naming conventions
        c1 = consec.get("benign", consec.get("HP", []))
        c2 = consec.get("malignant", consec.get("SSA", []))
        avg = [(a+b)/2 for a,b in zip(c1, c2)]
        ax_.plot(range(1, T), avg, **label_style(r), alpha=0.7, linewidth=1.3, markersize=5)
    ax_.axhline(0.998, color="gray", linestyle=":", alpha=0.5)
    ax_.set_xlabel("t → t+1"); ax_.set_title(title)
    ax_.set_ylim(0.9, 1.001); ax_.grid(alpha=0.3)
    ax_.legend(fontsize=7, loc="lower right", ncol=2)
axes[0].set_ylabel("consecutive-timestep feature cosine")
plt.suptitle("Cross-dataset Fig A — Temporal collapse replicates from BACH to MHIST\n"
             "features saturate by t≈4 on both datasets, all configurations, all seeds", y=1.03)
plt.tight_layout()
plt.savefig(os.path.join(CROSS_OUT, "crossFigA_temporal_collapse_both.png"), dpi=200, bbox_inches="tight")
plt.savefig(os.path.join(CROSS_OUT, "crossFigA_temporal_collapse_both.pdf"), bbox_inches="tight")
plt.show()

# ── Cross Fig B: Test AUC — paired comparison across datasets ─────────
fig, ax = plt.subplots(figsize=(9, 5.5))
dataset_x = {"BACH": 0, "MHIST": 1}
seeds = [42, 123, 2024]
seed_offset = {42: -0.12, 123: 0.0, 2024: +0.12}
for r in bach + mhist:
    cfg = r["cfg"]
    if cfg["T"] != 16 or cfg["experiment"] not in ("baseline_learn","matches"): continue
    aucs = r["npz"]["test_aucs"].item()
    ds_x = dataset_x[r["dataset"]]
    exp_offset = -0.25 if cfg["experiment"] == "baseline_learn" else 0.25
    x = ds_x + exp_offset + seed_offset[cfg["seed"]]
    ax.scatter(x, aucs[16], color=COLORS[cfg["experiment"]], marker=MARKERS[cfg["seed"]],
               s=90, alpha=0.85, edgecolor="black", linewidth=0.5)
# Group means
for ds, ds_x in dataset_x.items():
    for exp, exp_offset in [("baseline_learn", -0.25), ("matches", 0.25)]:
        vals = []
        for r in (bach if ds == "BACH" else mhist):
            if r["cfg"]["T"] == 16 and r["cfg"]["experiment"] == exp:
                vals.append(r["npz"]["test_aucs"].item()[16])
        m = np.mean(vals); s = np.std(vals)
        ax.errorbar(ds_x + exp_offset, m, yerr=s, color=COLORS[exp], marker="_",
                    markersize=22, markeredgewidth=3, capsize=8, elinewidth=2, zorder=0, alpha=0.5)
ax.set_xticks(list(dataset_x.values())); ax.set_xticklabels(list(dataset_x.keys()))
ax.set_ylabel("Test AUC at t=16")
ax.set_title("Cross-dataset Fig B — matches does not beat baseline_learn on either dataset\n"
             "MHIST paired Δ = +0.004 (p ≈ 0.75); BACH paired Δ = +0.012 (p ≈ 0.26)")
from matplotlib.patches import Patch
handles = [Patch(color=COLORS["baseline_learn"], label="baseline_learn"),
           Patch(color=COLORS["matches"], label="matches")]
ax.legend(handles=handles, loc="lower right")
ax.grid(alpha=0.3, axis="y"); ax.set_ylim(0.72, 0.88)
plt.tight_layout()
plt.savefig(os.path.join(CROSS_OUT, "crossFigB_test_auc_both.png"), dpi=200, bbox_inches="tight")
plt.savefig(os.path.join(CROSS_OUT, "crossFigB_test_auc_both.pdf"), bbox_inches="tight")
plt.show()

# ── Cross Fig C: Accuracy vs SynOps overlay (both datasets) ───────────
fig, ax = plt.subplots(figsize=(11, 6.5))
ann_macs = sum(bach[0]["npz"]["macs"].item().values())
ax.axvline(ann_macs, color="black", linestyle=":", alpha=0.8, linewidth=2,
           label=f"ANN forward pass ({ann_macs/1e8:.2f}×10⁸)")
for r in bach + mhist:
    if r["cfg"]["T"] != 16 or r["cfg"]["experiment"] not in ("baseline_learn","matches"): continue
    T = r["cfg"]["T"]
    total_cum = r["npz"]["synops_total_cum"]; aucs = r["npz"]["test_aucs"].item()
    ts_plot = [t for t in [1,4,8,12,16] if t <= T]
    xs = [total_cum[t-1] for t in ts_plot]; ys = [aucs[t] for t in ts_plot]
    exp = r["cfg"]["experiment"]
    ls = "-" if r["dataset"] == "MHIST" else "--"
    marker = "o" if r["dataset"] == "MHIST" else "s"
    alpha = 0.7 if r["dataset"] == "MHIST" else 0.4
    ax.plot(xs, ys, marker=marker, linestyle=ls, color=COLORS[exp], alpha=alpha,
            markersize=6, linewidth=1.2)
ax.set_xscale("log")
tick_locs = [1e8, 2e8, 3e8, 5e8, 7e8, 1e9, 2e9, 3e9, 5e9]
ax.set_xticks(tick_locs)
def fmt(v, pos): return f"{v/1e8:.0f}×10⁸" if v < 1e9 else f"{v/1e9:.0f}×10⁹"
ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt))
ax.set_xlim(1e8, 5e9)
ax.set_xlabel("Cumulative SynOps (log scale)"); ax.set_ylabel("Test AUC at timestep t")
ax.set_title("Cross-dataset Fig C — Waste-zone pattern is invariant to dataset:\n"
             "compute grows ~4× from t=4 to t=16 with marginal AUC gain on both datasets")
# Custom legend
from matplotlib.lines import Line2D
handles = [
    Line2D([0],[0], color="black", linestyle=":", label="ANN forward pass"),
    Line2D([0],[0], color=COLORS["baseline_learn"], marker="s", linestyle="--", label="BACH baseline_learn"),
    Line2D([0],[0], color=COLORS["matches"], marker="s", linestyle="--", label="BACH matches"),
    Line2D([0],[0], color=COLORS["baseline_learn"], marker="o", linestyle="-", label="MHIST baseline_learn"),
    Line2D([0],[0], color=COLORS["matches"], marker="o", linestyle="-", label="MHIST matches"),
]
ax.legend(handles=handles, fontsize=9, loc="lower right")
ax.grid(True, which="major", alpha=0.4); ax.grid(True, which="minor", alpha=0.15, linestyle=":")
plt.tight_layout()
plt.savefig(os.path.join(CROSS_OUT, "crossFigC_accuracy_vs_synops_both.png"), dpi=200, bbox_inches="tight")
plt.savefig(os.path.join(CROSS_OUT, "crossFigC_accuracy_vs_synops_both.pdf"), bbox_inches="tight")
plt.show()

# ── Cross Fig D: Intervention summary combined heatmap ────────────────
summary_rows = []
for r in bach + mhist:
    cfg = r["cfg"]; aucs = r["npz"]["test_aucs"].item(); probe = r["npz"]["probe"].item()
    consec = r["npz"]["consec"].item()
    c1 = consec.get("benign", consec.get("HP", []))
    c2 = consec.get("malignant", consec.get("SSA", []))
    late_cos = float(np.mean(c1[4:] + c2[4:])) if cfg["T"] > 5 else float("nan")
    dp = probe[max(probe.keys())][0] - probe[4][0] if 4 in probe and max(probe.keys()) > 4 else 0.0
    summary_rows.append({
        "config": f"{r['dataset']}_{cfg['experiment']}_s{cfg['seed']}_T{cfg['T']}",
        "test_auc": aucs[cfg["T"]], "probe_final": probe[max(probe.keys())][0],
        "late_cos": late_cos, "delta_probe_t4_to_T": dp,
    })
df_sum = pd.DataFrame(summary_rows).set_index("config")
fig, ax = plt.subplots(figsize=(10, max(3, 0.4 * len(df_sum))))
im = ax.imshow(df_sum.values, cmap="RdYlGn", aspect="auto")
ax.set_xticks(range(len(df_sum.columns))); ax.set_xticklabels(df_sum.columns, rotation=15, ha="right")
ax.set_yticks(range(len(df_sum.index))); ax.set_yticklabels(df_sum.index, fontsize=8)
for i in range(df_sum.shape[0]):
    for j in range(df_sum.shape[1]):
        ax.text(j, i, f"{df_sum.values[i,j]:.3f}", ha="center", va="center", fontsize=7, color="black")
plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
ax.set_title("Cross-dataset Fig D — Intervention summary (BACH + MHIST, all 14 runs)")
plt.tight_layout()
plt.savefig(os.path.join(CROSS_OUT, "crossFigD_intervention_summary_both.png"), dpi=200, bbox_inches="tight")
plt.savefig(os.path.join(CROSS_OUT, "crossFigD_intervention_summary_both.pdf"), bbox_inches="tight")
plt.show()

# ══════════════════════════════════════════════════════════════════════
# COMBINED TABLES (both datasets)
# ══════════════════════════════════════════════════════════════════════

# Table 5 (extended): per-run main results
table5 = []
for r in bach + mhist:
    cfg = r["cfg"]; aucs = r["npz"]["test_aucs"].item(); probe = r["npz"]["probe"].item()
    lo, med, hi = r["npz"]["boot_ci"]
    table5.append({
        "dataset": r["dataset"], "experiment": cfg["experiment"], "seed": cfg["seed"], "T": cfg["T"],
        "test_auc_final": aucs[cfg["T"]], "boot_lo": float(lo), "boot_hi": float(hi),
        "probe_final_mean": probe[max(probe.keys())][0], "probe_final_std": probe[max(probe.keys())][1],
    })
pd.DataFrame(table5).to_csv(os.path.join(CROSS_OUT, "table5_main_results_both.csv"), index=False)

# Table 6 (extended): paired matches − baseline_learn per seed, per dataset
def paired_table(runs, dataset):
    bl = {r["cfg"]["seed"]: r for r in runs if r["cfg"]["experiment"] == "baseline_learn" and r["cfg"]["T"] == 16}
    mt = {r["cfg"]["seed"]: r for r in runs if r["cfg"]["experiment"] == "matches" and r["cfg"]["T"] == 16}
    out = []
    for s in sorted(set(bl.keys()) & set(mt.keys())):
        b_auc = bl[s]["npz"]["test_aucs"].item()[16]; m_auc = mt[s]["npz"]["test_aucs"].item()[16]
        b_probe = bl[s]["npz"]["probe"].item()[16][0]; m_probe = mt[s]["npz"]["probe"].item()[16][0]
        out.append({"dataset": dataset, "seed": s,
                    "baseline_test": b_auc, "matches_test": m_auc, "delta_test": m_auc - b_auc,
                    "baseline_probe": b_probe, "matches_probe": m_probe, "delta_probe": m_probe - b_probe})
    return out
table6 = paired_table(bach, "BACH") + paired_table(mhist, "MHIST")
pd.DataFrame(table6).to_csv(os.path.join(CROSS_OUT, "table6_paired_by_seed_both.csv"), index=False)

# Table 7 (extended): statistical tests per dataset
def paired_permutation(a, b, n=10000, seed=0):
    a = np.asarray(a); b = np.asarray(b); obs = np.mean(a - b)
    rng = np.random.default_rng(seed); cnt = 0
    for _ in range(n):
        signs = rng.choice([-1, 1], size=len(a))
        if abs(np.mean(signs * (a - b))) >= abs(obs): cnt += 1
    return obs, cnt / n
def cohens_d_paired(a, b):
    d = np.asarray(a) - np.asarray(b); return float(np.mean(d) / (np.std(d, ddof=1) + 1e-12))
stat_rows = []
for ds, ds_rows in [("BACH", [r for r in table6 if r["dataset"] == "BACH"]),
                    ("MHIST", [r for r in table6 if r["dataset"] == "MHIST"])]:
    if not ds_rows: continue
    a_t = [r["matches_test"] for r in ds_rows]; b_t = [r["baseline_test"] for r in ds_rows]
    a_p = [r["matches_probe"] for r in ds_rows]; b_p = [r["baseline_probe"] for r in ds_rows]
    obs_t, p_t = paired_permutation(a_t, b_t); obs_p, p_p = paired_permutation(a_p, b_p)
    stat_rows.append({"dataset": ds, "comparison": "matches − baseline_learn TEST",
                      "n": len(a_t), "mean_diff": obs_t, "p_perm": p_t,
                      "cohens_d_paired": cohens_d_paired(a_t, b_t)})
    stat_rows.append({"dataset": ds, "comparison": "matches − baseline_learn PROBE",
                      "n": len(a_p), "mean_diff": obs_p, "p_perm": p_p,
                      "cohens_d_paired": cohens_d_paired(a_p, b_p)})
pd.DataFrame(stat_rows).to_csv(os.path.join(CROSS_OUT, "table7_statistical_tests_both.csv"), index=False)

# Table 8 (extended): SynOps ablation
table8 = []
for r in bach + mhist:
    cfg = r["cfg"]; aucs = r["npz"]["test_aucs"].item(); total_cum = r["npz"]["synops_total_cum"]
    row = {"dataset": r["dataset"], "config": f"{cfg['experiment']}_s{cfg['seed']}_T{cfg['T']}"}
    for t in [1, 4, 8, 12, 16]:
        if t <= cfg["T"]:
            row[f"synops_t{t}"] = float(total_cum[t-1]); row[f"auc_t{t}"] = aucs.get(t, float("nan"))
    if cfg["T"] == 16:
        row["synops_ratio_t16_t4"] = float(total_cum[15] / total_cum[3])
        row["auc_gain_t4_to_t16"]  = aucs[16] - aucs[4]
    table8.append(row)
pd.DataFrame(table8).to_csv(os.path.join(CROSS_OUT, "table8_ablation_synops_both.csv"), index=False)

# ── Print quick summary ───────────────────────────────────────────────
print("\n" + "="*70)
print(" CROSS-DATASET SUMMARY")
print("="*70)
df7 = pd.DataFrame(stat_rows)
print(df7.to_string(index=False))
print()

print("\nFigures written:")
for d in [MHIST_OUT, CROSS_OUT]:
    print(f"\n  {d}/")
    for f in sorted(os.listdir(d)):
        print(f"    {f}")