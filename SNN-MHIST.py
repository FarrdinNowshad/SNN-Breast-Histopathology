!pip install -q snntorch

import os, json, random, warnings, time, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import snntorch as snn
from snntorch import surrogate

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# ─── ACCOUNT-SPECIFIC CONFIG ───────────────────────────────────────────

if os.path.isdir("/kaggle/input/datasets/farrdinnowshad/mhist-6-2-2"):
    ACCOUNT = "farrdinnowshad"
    DATA_ROOT = "/kaggle/input/datasets/farrdinnowshad/mhist-6-2-2"
    RUN_MATRIX = [
        {"experiment": "baseline_learn", "seed": 42,   "T": 16},
        {"experiment": "baseline_learn", "seed": 123,  "T": 16},
        {"experiment": "baseline_learn", "seed": 2024, "T": 16},
    ]
elif os.path.isdir("/kaggle/input/datasets/preanto/mhist-6-2-2"):
    ACCOUNT = "preanto"
    DATA_ROOT = "/kaggle/input/datasets/preanto/mhist-6-2-2"
    RUN_MATRIX = [
        {"experiment": "matches", "seed": 42,   "T": 16},
        {"experiment": "matches", "seed": 123,  "T": 16},
        {"experiment": "matches", "seed": 2024, "T": 16},
    ]
else:
    raise RuntimeError("Neither farrdinnowshad's nor preanto's MHIST dataset is attached. Check dataset attachments.")

print(f"auto-detected account: {ACCOUNT}")

# ─── EXPERIMENT REGISTRY ───────────────────────────────────────────────
EXPERIMENTS = {
    "baseline_learn": {
        "learn_beta": True,
        "schedule": {"stem": 0.95, "s1_lif1": 0.95, "s1_lif2": 0.95,
                     "s2_lif1": 0.95, "s2_lif2": 0.95,
                     "s3_lif1": 0.95, "s3_lif2": 0.95,
                     "s4_lif1": 0.95, "s4_lif2": 0.95},
        "notes": "uniform β=0.95, learn_beta=True",
    },
    "matches": {
        "learn_beta": False,
        "schedule": {"stem": 1.00,
                     "s1_lif1": 0.90, "s1_lif2": 0.98,
                     "s2_lif1": 0.93, "s2_lif2": 0.97,
                     "s3_lif1": 0.92, "s3_lif2": 0.96,
                     "s4_lif1": 0.90, "s4_lif2": 0.94},
        "notes": "matches_preference — main positive result candidate",
    },
}

# ─── FIXED CONFIG (same as BACH for cross-dataset comparability) ───────
IMG_SIZE       = 160         # keep same as BACH → SpikingResNetLite feature maps identical
BATCH          = 24
EPOCHS         = 120
LR             = 1e-3
WD             = 5e-4
GRAD_CLIP      = 1.0
PATIENCE       = 75
BETA_LOG_EVERY = 5
POS_WEIGHT_VAL = 1236.0 / 504.0   # = 2.452, from train split HP:SSA ratio

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WORK_ROOT = "/kaggle/working"

print(f"account: {ACCOUNT}")
print(f"device:  {DEVICE}")
print(f"cuda:    {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
print(f"runs:    {len(RUN_MATRIX)}")
print(f"pos_weight: {POS_WEIGHT_VAL:.4f} (upweights SSA errors)")

# ─── MODEL ─────────────────────────────────────────────────────────────
spike_grad = surrogate.atan()
def make_lif(beta, learn_beta):
    return snn.Leaky(beta=beta, spike_grad=spike_grad,
                     init_hidden=False, learn_beta=learn_beta)

class SpikingResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride, beta1, beta2, learn_beta):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_c)
        self.lif1  = make_lif(beta1, learn_beta)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_c)
        self.lif2  = make_lif(beta2, learn_beta)
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c))
        else:
            self.shortcut = nn.Identity()
    def forward(self, x, m1, m2):
        sc = self.shortcut(x)
        s1, m1 = self.lif1(self.bn1(self.conv1(x)), m1)
        s2, m2 = self.lif2(self.bn2(self.conv2(s1)) + sc, m2)
        return s2, m1, m2

class SpikingResNetLite(nn.Module):
    def __init__(self, T, betas, learn_beta):
        super().__init__()
        self.T = T
        self.stem_conv = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.stem_bn   = nn.BatchNorm2d(64)
        self.stem_lif  = make_lif(betas["stem"], learn_beta)
        self.stem_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.stage1 = SpikingResBlock(64,  64,  1, betas["s1_lif1"], betas["s1_lif2"], learn_beta)
        self.stage2 = SpikingResBlock(64,  128, 2, betas["s2_lif1"], betas["s2_lif2"], learn_beta)
        self.stage3 = SpikingResBlock(128, 256, 2, betas["s3_lif1"], betas["s3_lif2"], learn_beta)
        self.stage4 = SpikingResBlock(256, 512, 2, betas["s4_lif1"], betas["s4_lif2"], learn_beta)
        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc      = nn.Linear(512, 1)
    def forward(self, x, return_features=False):
        ms  = self.stem_lif.init_leaky()
        m1a = self.stage1.lif1.init_leaky(); m1b = self.stage1.lif2.init_leaky()
        m2a = self.stage2.lif1.init_leaky(); m2b = self.stage2.lif2.init_leaky()
        m3a = self.stage3.lif1.init_leaky(); m3b = self.stage3.lif2.init_leaky()
        m4a = self.stage4.lif1.init_leaky(); m4b = self.stage4.lif2.init_leaky()
        step_logits = []
        features = [] if return_features else None
        for t in range(self.T):
            s_stem, ms = self.stem_lif(self.stem_bn(self.stem_conv(x)), ms)
            s_stem = self.stem_pool(s_stem)
            s1, m1a, m1b = self.stage1(s_stem, m1a, m1b)
            s2, m2a, m2b = self.stage2(s1,     m2a, m2b)
            s3, m3a, m3b = self.stage3(s2,     m3a, m3b)
            s4, m4a, m4b = self.stage4(s3,     m4a, m4b)
            pooled = self.gap(s4).flatten(1)
            step_logits.append(self.fc(self.dropout(pooled)).squeeze(1))
            if return_features:
                features.append(pooled)
        step_logits = torch.stack(step_logits, dim=0)
        cum = torch.cumsum(step_logits, dim=0) / \
              torch.arange(1, self.T+1, device=x.device).unsqueeze(1).float()
        if return_features:
            return step_logits, cum, torch.stack(features, dim=0)
        return step_logits, cum

# ─── DATA — MHIST folder structure, HP/SSA classes ─────────────────────
IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
def load_split(root, split_name, class_names=("HP", "SSA")):
    samples = []
    for cls_idx, cls in enumerate(class_names):
        d = os.path.join(root, split_name, cls)
        if not os.path.isdir(d):
            raise RuntimeError(f"missing directory: {d}")
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(IMG_EXT):
                samples.append((os.path.join(d, f), cls_idx))
    return samples

class MHISTSubset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        p, y = self.samples[i]
        return self.transform(Image.open(p).convert("RGB")), torch.tensor(y, dtype=torch.float32)

imagenet_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_tf = transforms.Compose([
    transforms.Resize((int(IMG_SIZE*1.1), int(IMG_SIZE*1.1))),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.3, 0.3, 0.2, 0.05),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(), imagenet_norm,
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15))])
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(), imagenet_norm])

train_samples = load_split(DATA_ROOT, "train")
val_samples   = load_split(DATA_ROOT, "val")
test_samples  = load_split(DATA_ROOT, "test")

def counts(s):
    lab = [y for _, y in s]
    return len(s), lab.count(0), lab.count(1)   # HP=0, SSA=1
print(f"\nsplits — train={counts(train_samples)} (n, HP, SSA)")
print(f"         val  ={counts(val_samples)}")
print(f"         test ={counts(test_samples)}")

# Assertions match the exact counts from your preprocessing verification
assert counts(train_samples) == (1740, 1236, 504), f"train count mismatch: {counts(train_samples)}"
assert counts(val_samples)   == (435,  309,  126), f"val count mismatch: {counts(val_samples)}"
assert counts(test_samples)  == (977,  617,  360), f"test count mismatch: {counts(test_samples)}"
print("✓ split counts verified\n")

# Overlap assertions
tp = {p for p,_ in train_samples}; vp = {p for p,_ in val_samples}; tsp = {p for p,_ in test_samples}
assert len(tp & vp) == 0 and len(tp & tsp) == 0 and len(vp & tsp) == 0
print("✓ no cross-split overlap")

# ─── ANALYSIS HELPERS (identical to BACH master script) ────────────────
def snapshot_betas(model):
    return {name + ".beta":
            (m.beta.item() if isinstance(m.beta, torch.Tensor) else float(m.beta))
            for name, m in model.named_modules() if isinstance(m, snn.Leaky)}

def cos_sim(a, b):
    a = a / (np.linalg.norm(a) + 1e-9); b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))

def compute_static_macs(model):
    macs, hooks = {}, []
    def make_hook(name):
        def hook(mod, inp, out):
            k_h, k_w = mod.kernel_size
            c_in, c_out = inp[0].shape[1], out.shape[1]
            oh, ow = out.shape[2], out.shape[3]
            macs[name] = int(oh * ow * c_out * c_in * k_h * k_w)
        return hook
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(make_hook(n)))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE))
    for h in hooks: h.remove()
    return macs

def collect_spike_rates(model, loader, T):
    per_t_sum   = {n: np.zeros(T) for n, m in model.named_modules() if isinstance(m, snn.Leaky)}
    per_t_count = {n: np.zeros(T) for n in per_t_sum}
    call_idx    = {n: 0 for n in per_t_sum}
    def make_hook(name):
        def hook(mod, inp, out):
            spk = out[0] if isinstance(out, tuple) else out
            rate = spk.float().mean().item()
            bs = spk.shape[0]
            t = call_idx[name] % T
            per_t_sum[name][t]   += rate * bs
            per_t_count[name][t] += bs
            call_idx[name] += 1
        return hook
    hooks = []
    for n, m in model.named_modules():
        if isinstance(m, snn.Leaky):
            hooks.append(m.register_forward_hook(make_hook(n)))
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            model(x.to(DEVICE))
    for h in hooks: h.remove()
    return {n: per_t_sum[n] / np.maximum(per_t_count[n], 1) for n in per_t_sum}

LIF_TO_CONVS = {
    "stem_lif":    ["stage1.conv1"],
    "stage1.lif1": ["stage1.conv2"],
    "stage1.lif2": ["stage2.conv1", "stage2.shortcut.0"],
    "stage2.lif1": ["stage2.conv2"],
    "stage2.lif2": ["stage3.conv1", "stage3.shortcut.0"],
    "stage3.lif1": ["stage3.conv2"],
    "stage3.lif2": ["stage4.conv1", "stage4.shortcut.0"],
    "stage4.lif1": ["stage4.conv2"],
}
def compute_synops(spike_rates, macs, T):
    per_conv_per_t = {"stem_conv": macs["stem_conv"] * np.ones(T)}
    for lif_name, conv_names in LIF_TO_CONVS.items():
        rate_t = spike_rates[lif_name]
        for cn in conv_names:
            if cn in macs:
                per_conv_per_t[cn] = macs[cn] * rate_t
    by_layer_cum = {cn: np.cumsum(v) for cn, v in per_conv_per_t.items()}
    total_cum = np.sum(list(by_layer_cum.values()), axis=0)
    return per_conv_per_t, total_cum, by_layer_cum

# ─── LOSS with pos_weight ──────────────────────────────────────────────
def anytime_weights(T, device):
    w = torch.tensor([T + t for t in range(1, T+1)], dtype=torch.float32, device=device)
    return w / w.sum()

def anytime_loss(cum_logit, y, W_T, pos_weight):
    per_t = torch.stack([
        F.binary_cross_entropy_with_logits(cum_logit[t], y, pos_weight=pos_weight)
        for t in range(cum_logit.shape[0])
    ])
    return (per_t * W_T).sum()

def eval_loader(model, loader, T):
    model.eval()
    probs_t = {t: [] for t in [1, 4, 8, 12, 16] if t <= T}
    ys = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            _, cum = model(x)
            for t in probs_t:
                probs_t[t].append(torch.sigmoid(cum[t-1]).cpu().numpy())
            ys.append(y.numpy())
    ys = np.concatenate(ys)
    aucs = {t: roc_auc_score(ys, np.concatenate(probs_t[t])) if len(np.unique(ys)) > 1 else float("nan")
            for t in probs_t}
    return aucs, ys

def cv_probe(features_t, labels, n_folds=5, seed=0):
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    aucs = []
    for tr, va in skf.split(features_t, labels):
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        clf.fit(features_t[tr], labels[tr])
        p = clf.predict_proba(features_t[va])[:, 1]
        if len(np.unique(labels[va])) > 1:
            aucs.append(roc_auc_score(labels[va], p))
    return float(np.mean(aucs)), float(np.std(aucs))

def bootstrap_auc(probs, ys, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    aucs = []
    N = len(ys)
    for _ in range(n):
        idx = rng.integers(0, N, N)
        if len(np.unique(ys[idx])) < 2: continue
        aucs.append(roc_auc_score(ys[idx], probs[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 50)), float(np.percentile(aucs, 97.5))

# ─── TRAINING FUNCTION ─────────────────────────────────────────────────
def train_one(cfg):
    exp_name, seed, T = cfg["experiment"], cfg["seed"], cfg["T"]
    exp_cfg  = EXPERIMENTS[exp_name]
    schedule, learn_beta = exp_cfg["schedule"], exp_cfg["learn_beta"]
    out_dir  = os.path.join(WORK_ROOT, f"mhist_{exp_name}_T{T}_seed{seed}")
    os.makedirs(out_dir, exist_ok=True)

    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    print("="*70)
    print(f"  {exp_name}  seed={seed}  T={T}  ({exp_cfg['notes']})")
    print("="*70)

    g = torch.Generator(); g.manual_seed(seed)
    def _wi(wid): np.random.seed(seed + wid); random.seed(seed + wid)
    train_ds = MHISTSubset(train_samples, train_tf)
    val_ds   = MHISTSubset(val_samples,   eval_tf)
    test_ds  = MHISTSubset(test_samples,  eval_tf)
    train_ld = DataLoader(train_ds, BATCH, shuffle=True, num_workers=4, pin_memory=True,
                          drop_last=True, generator=g, worker_init_fn=_wi)
    val_ld   = DataLoader(val_ds,   BATCH, shuffle=False, num_workers=4, pin_memory=True, worker_init_fn=_wi)
    test_ld  = DataLoader(test_ds,  BATCH, shuffle=False, num_workers=4, pin_memory=True, worker_init_fn=_wi)

    model = SpikingResNetLite(T=T, betas=schedule, learn_beta=learn_beta).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    W_T   = anytime_weights(T, DEVICE)
    pos_weight = torch.tensor([POS_WEIGHT_VAL], dtype=torch.float32).to(DEVICE)

    with open(os.path.join(out_dir, "sidecar.json"), "w") as f:
        json.dump({
            "experiment": exp_name, "dataset": "MHIST", "notes": exp_cfg["notes"],
            "learn_beta": learn_beta, "schedule": schedule,
            "T": T, "seed": seed, "account": ACCOUNT,
            "split_source": "MHIST official test intact; train 80/20 stratified into train/val",
            "img_size": IMG_SIZE, "batch": BATCH, "pos_weight": POS_WEIGHT_VAL,
            "epochs": EPOCHS, "lr": LR, "wd": WD, "patience": PATIENCE, "determinism": True,
        }, f, indent=2)

    best_val_auc, best_state, since_best = 0.0, None, 0
    history = []
    beta_history = [{"epoch": 0, **snapshot_betas(model)}]
    t_start = time.time()

    for ep in range(1, EPOCHS + 1):
        model.train()
        ep_loss, n = 0.0, 0
        t0 = time.time()
        for x, y in train_ld:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            _, cum = model(x)
            loss = anytime_loss(cum, y, W_T, pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            ep_loss += loss.item() * x.size(0); n += x.size(0)
        sched.step()
        ep_loss /= max(n, 1)

        val_aucs, _ = eval_loader(model, val_ld, T)
        val_key = T if T in val_aucs else max(val_aucs.keys())
        history.append({"epoch": ep, "loss": ep_loss,
                        **{f"val_auc_t{t}": v for t, v in val_aucs.items()}})

        msg = f"ep {ep:03d}  loss={ep_loss:.4f}  " + \
              "  ".join(f"val_auc(t={t})={v:.3f}" for t, v in val_aucs.items()) + \
              f"  ({time.time()-t0:.1f}s)"
        if val_aucs[val_key] > best_val_auc:
            best_val_auc = val_aucs[val_key]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            since_best = 0
            msg += "  ★"
        else:
            since_best += 1
        if ep % BETA_LOG_EVERY == 0 or ep == EPOCHS:
            beta_history.append({"epoch": ep, **snapshot_betas(model)})
        if ep % 5 == 0 or ep <= 3 or ep >= EPOCHS - 3:
            print(msg)
        if since_best >= PATIENCE:
            print(f"early stop @ ep {ep}")
            break
    print(f"training done in {(time.time()-t_start)/60:.1f} min")

    # Final eval
    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(out_dir, "best_val_state.pt"))
    pd.DataFrame(history).to_csv(os.path.join(out_dir, "history.csv"), index=False)
    pd.DataFrame(beta_history).to_csv(os.path.join(out_dir, "beta_history.csv"), index=False)

    print(f"\nfinal test-set evaluation:")
    test_aucs, test_ys = eval_loader(model, test_ld, T)
    for t, v in test_aucs.items():
        print(f"  test AUC (t={t:2d}) = {v:.4f}")

    @torch.no_grad()
    def extract_features(loader):
        model.eval()
        feats, labels_arr = [], []
        for x, y in loader:
            x = x.to(DEVICE)
            _, _, f = model(x, return_features=True)
            feats.append(f.cpu().numpy())
            labels_arr.extend(y.numpy().tolist())
        return np.concatenate(feats, axis=1), np.array(labels_arr)
    features, labels = extract_features(test_ld)

    with torch.no_grad():
        probs_final, ys_final = [], []
        for x, y in test_ld:
            x = x.to(DEVICE)
            _, cum = model(x)
            probs_final.append(torch.sigmoid(cum[T-1]).cpu().numpy())
            ys_final.append(y.numpy())
    probs_final = np.concatenate(probs_final); ys_final = np.concatenate(ys_final)
    lo, med, hi = bootstrap_auc(probs_final, ys_final, n=1000, seed=seed)
    print(f"bootstrap test AUC(t={T})  median={med:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]")

    probe = {}
    for t in [t for t in [1, 4, 8, 12, 16] if t <= T]:
        m, s = cv_probe(features[t-1], labels, n_folds=5, seed=seed)
        probe[t] = (m, s)
        print(f"  probe AUC t={t:2d}  {m:.3f} ± {s:.3f}")

    consec = {"HP": [], "SSA": []}
    for cls_idx, cls_name in enumerate(["HP", "SSA"]):
        mask = labels == cls_idx
        traj = features[:, mask].mean(axis=1)
        for ti in range(T - 1):
            consec[cls_name].append(cos_sim(traj[ti], traj[ti+1]))

    pairwise_cos = {}
    for cls_idx, cls_name in enumerate(["HP", "SSA"]):
        mask = labels == cls_idx
        traj = features[:, mask].mean(axis=1)
        mat = np.zeros((T, T))
        for i in range(T):
            for j in range(T):
                mat[i, j] = cos_sim(traj[i], traj[j])
        pairwise_cos[cls_name] = mat

    macs = compute_static_macs(model)
    spike_rates = collect_spike_rates(model, test_ld, T)
    per_conv, total_cum, by_layer_cum = compute_synops(spike_rates, macs, T)
    final_betas = snapshot_betas(model)

    np.savez(os.path.join(out_dir, "analysis.npz"),
             features=features, labels=labels, probe=probe, consec=consec,
             pairwise_cos_HP=pairwise_cos["HP"],
             pairwise_cos_SSA=pairwise_cos["SSA"],
             test_aucs=test_aucs, boot_ci=(lo, med, hi),
             final_betas=final_betas,
             spike_rates=spike_rates, macs=macs,
             synops_per_conv_per_t=per_conv,
             synops_total_cum=total_cum,
             synops_by_layer_cum=by_layer_cum,
             probs_final_test=probs_final, ys_final_test=ys_final)

    print(f"\n✓ artifacts → {out_dir}")
    del model, opt, sched, best_state, features, probs_final
    gc.collect()
    torch.cuda.empty_cache()
    return out_dir

# ─── MAIN LOOP ─────────────────────────────────────────────────────────
completed, failed = [], []
overall_start = time.time()
for i, cfg in enumerate(RUN_MATRIX, 1):
    print(f"\n\n{'#'*72}\n#  RUN {i}/{len(RUN_MATRIX)}  |  elapsed: {(time.time()-overall_start)/60:.1f} min\n{'#'*72}")
    try:
        od = train_one(cfg)
        completed.append((cfg, od))
    except Exception as e:
        print(f"\n!!! FAILED: {cfg} — {type(e).__name__}: {e}")
        failed.append((cfg, str(e)))
        gc.collect()
        torch.cuda.empty_cache()

print(f"\n\n{'='*72}\nDONE — {len(completed)} completed, {len(failed)} failed  ({(time.time()-overall_start)/60:.1f} min total)\n{'='*72}")
for cfg, od in completed:
    print(f"  ✓ {cfg['experiment']:>16s} s{cfg['seed']:>4} T{cfg['T']:>2}  → {od}")
for cfg, err in failed:
    print(f"  ✗ {cfg['experiment']:>16s} s{cfg['seed']:>4} T{cfg['T']:>2}  ({err[:80]})")

# Quick verification the output dirs are populated
print("\n\nOutput inventory (for verifying persistence after commit):")
for cfg, od in completed:
    for f in ["best_val_state.pt", "sidecar.json", "analysis.npz", "history.csv", "beta_history.csv"]:
        exists = os.path.exists(os.path.join(od, f))
        sz = os.path.getsize(os.path.join(od, f)) / 1e6 if exists else 0
        print(f"  {os.path.basename(od)}/{f}  {'✓' if exists else '✗'}  ({sz:.1f} MB)")