!pip install -q snntorch

import os, json, random, warnings, time, gc, itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
try:
    from sklearn.manifold import TSNE
    HAS_TSNE = True
except Exception:
    HAS_TSNE = False

import snntorch as snn
from snntorch import surrogate

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# ╔══════════════════════════════════════════════════════════════════════╗
# ║ RUN MATRIX                                                            ║
# ╚══════════════════════════════════════════════════════════════════════╝
RUN_MATRIX = [
    {"experiment": "baseline_learn",  "seed": 42,   "T": 16},
    {"experiment": "baseline_learn",  "seed": 123,  "T": 16},
    {"experiment": "baseline_learn",  "seed": 2024, "T": 16},
    {"experiment": "matches",         "seed": 42,   "T": 16},
    {"experiment": "matches",         "seed": 123,  "T": 16},
    {"experiment": "matches",         "seed": 2024, "T": 16},
    {"experiment": "baseline_frozen", "seed": 42,   "T": 16},
    {"experiment": "baseline_learn",  "seed": 42,   "T": 4},
]

# ╔══════════════════════════════════════════════════════════════════════╗
# ║ EXPERIMENT REGISTRY                                                   ║
# ╚══════════════════════════════════════════════════════════════════════╝
EXPERIMENTS = {
    "baseline_learn": {
        "learn_beta": True,
        "schedule": {"stem": 0.95, "s1_lif1": 0.95, "s1_lif2": 0.95,
                     "s2_lif1": 0.95, "s2_lif2": 0.95,
                     "s3_lif1": 0.95, "s3_lif2": 0.95,
                     "s4_lif1": 0.95, "s4_lif2": 0.95},
        "notes": "uniform β=0.95, learn_beta=True",
    },
    "baseline_frozen": {
        "learn_beta": False,
        "schedule": {"stem": 0.95, "s1_lif1": 0.95, "s1_lif2": 0.95,
                     "s2_lif1": 0.95, "s2_lif2": 0.95,
                     "s3_lif1": 0.95, "s3_lif2": 0.95,
                     "s4_lif1": 0.95, "s4_lif2": 0.95},
        "notes": "learn_beta=False — isolates schedule vs freezing",
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

# ╔══════════════════════════════════════════════════════════════════════╗
# ║ FIXED CONFIG                                                          ║
# ╚══════════════════════════════════════════════════════════════════════╝
DATA_ROOT      = "/kaggle/input/datasets/preanto/bach-new"
IMG_SIZE       = 160
BATCH          = 24
EPOCHS         = 120
LR             = 1e-3
WD             = 5e-4
GRAD_CLIP      = 1.0
PATIENCE       = 75
BETA_LOG_EVERY = 5           # snapshot β every N epochs (learn_beta runs only need this)
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WORK_ROOT      = "/kaggle/working"
ARTIFACTS_DIR  = os.path.join(WORK_ROOT, "paper_artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

print(f"device: {DEVICE}")
print(f"CUDA:   {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
print(f"runs:   {len(RUN_MATRIX)}")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║ MODEL                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════╝
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

# ╔══════════════════════════════════════════════════════════════════════╗
# ║ DATA — Preanto 60/20/20 pre-split                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝
IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
def load_split(root, split_name):
    samples = []
    for cls_idx, cls in enumerate(["benign", "malignant"]):
        d = os.path.join(root, split_name, cls)
        if not os.path.isdir(d):
            raise RuntimeError(f"missing directory: {d}")
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(IMG_EXT):
                samples.append((os.path.join(d, f), cls_idx))
    return samples

class BACHSubset(Dataset):
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

# Overlap + balance assertions
tp = {p for p,_ in train_samples}; vp = {p for p,_ in val_samples}; tsp = {p for p,_ in test_samples}
assert len(tp & vp) == 0 and len(tp & tsp) == 0 and len(vp & tsp) == 0
def counts(s):
    lab = [y for _,y in s]
    return len(s), lab.count(0), lab.count(1)
assert counts(train_samples) == (240, 120, 120)
assert counts(val_samples)   == (80,  40,  40)
assert counts(test_samples)  == (80,  40,  40)
print(f"splits verified — train={counts(train_samples)}  val={counts(val_samples)}  test={counts(test_samples)}\n")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║ ANALYSIS HELPERS                                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝
def snapshot_betas(model):
    return {name + ".beta":
            (m.beta.item() if isinstance(m.beta, torch.Tensor) else float(m.beta))
            for name, m in model.named_modules() if isinstance(m, snn.Leaky)}

def cos_sim(a, b):
    a = a / (np.linalg.norm(a) + 1e-9); b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))

def compute_static_macs(model):
    """Per-conv static MACs (input-size dependent, computed via a zero forward)."""
    macs = {}
    hooks = []
    def make_hook(name):
        def hook(mod, inp, out):
            k_h, k_w = mod.kernel_size
            c_in  = inp[0].shape[1]
            c_out = out.shape[1]
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
    """Mean per-layer spike rate across full test set, weighted by batch size."""
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

# LIF → conv wiring for SynOps (stem_conv sees raw pixels, always fires)
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

# ╔══════════════════════════════════════════════════════════════════════╗
# ║ TRAINING FUNCTION — one config → per-run OUT_DIR                     ║
# ╚══════════════════════════════════════════════════════════════════════╝
def anytime_weights(T, device):
    w = torch.tensor([T + t for t in range(1, T+1)], dtype=torch.float32, device=device)
    return w / w.sum()

def anytime_loss(cum_logit, y, W_T):
    per_t = torch.stack([
        F.binary_cross_entropy_with_logits(cum_logit[t], y) for t in range(cum_logit.shape[0])
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
    aucs = {}
    for t in probs_t:
        aucs[t] = roc_auc_score(ys, np.concatenate(probs_t[t])) if len(np.unique(ys)) > 1 else float("nan")
    return aucs, ys

def cv_probe(features_t, labels, n_folds=5, seed=0):
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    aucs = []
    for tr, va in skf.split(features_t, labels):
        clf = LogisticRegression(max_iter=2000, C=1.0)
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

def train_one(cfg):
    """cfg = {'experiment': str, 'seed': int, 'T': int}. Returns out_dir path on success."""
    exp_name = cfg["experiment"]; seed = cfg["seed"]; T = cfg["T"]
    exp_cfg  = EXPERIMENTS[exp_name]
    schedule = exp_cfg["schedule"]; learn_beta = exp_cfg["learn_beta"]
    out_dir  = os.path.join(WORK_ROOT, f"bach_{exp_name}_T{T}_seed{seed}")
    os.makedirs(out_dir, exist_ok=True)

    # Seed
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    print("="*70)
    print(f"  {exp_name}  seed={seed}  T={T}  ({exp_cfg['notes']})")
    print("="*70)

    # Data
    g = torch.Generator(); g.manual_seed(seed)
    def _wi(wid): np.random.seed(seed + wid); random.seed(seed + wid)
    train_ds = BACHSubset(train_samples, train_tf)
    val_ds   = BACHSubset(val_samples,   eval_tf)
    test_ds  = BACHSubset(test_samples,  eval_tf)
    train_ld = DataLoader(train_ds, BATCH, shuffle=True,  num_workers=2, pin_memory=True,
                          drop_last=True, generator=g, worker_init_fn=_wi)
    val_ld   = DataLoader(val_ds,   BATCH, shuffle=False, num_workers=2, pin_memory=True, worker_init_fn=_wi)
    test_ld  = DataLoader(test_ds,  BATCH, shuffle=False, num_workers=2, pin_memory=True, worker_init_fn=_wi)

    # Model, opt, loss
    model = SpikingResNetLite(T=T, betas=schedule, learn_beta=learn_beta).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    W_T   = anytime_weights(T, DEVICE)

    # Sidecar
    with open(os.path.join(out_dir, "sidecar.json"), "w") as f:
        json.dump({
            "experiment": exp_name, "dataset": "BACH", "notes": exp_cfg["notes"],
            "learn_beta": learn_beta, "schedule": schedule,
            "T": T, "seed": seed, "split_source": "preanto/bach-new pre-defined 60/20/20 stratified",
            "img_size": IMG_SIZE, "batch": BATCH,
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
            loss = anytime_loss(cum, y, W_T)
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
        # Log β periodically (only useful for learn_beta=True but harmless otherwise)
        if ep % BETA_LOG_EVERY == 0 or ep == EPOCHS:
            beta_history.append({"epoch": ep, **snapshot_betas(model)})
        if ep % 10 == 0 or ep <= 5 or ep >= EPOCHS - 3:
            print(msg)
        if since_best >= PATIENCE:
            print(f"early stop @ ep {ep}")
            break
    print(f"training done in {(time.time()-t_start)/60:.1f} min")

    # Final eval on val-selected checkpoint
    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(out_dir, "best_val_state.pt"))
    pd.DataFrame(history).to_csv(os.path.join(out_dir, "history.csv"), index=False)
    pd.DataFrame(beta_history).to_csv(os.path.join(out_dir, "beta_history.csv"), index=False)

    print(f"\nfinal test-set evaluation (val-selected checkpoint):")
    test_aucs, test_ys = eval_loader(model, test_ld, T)
    for t, v in test_aucs.items():
        print(f"  test AUC (t={t:2d}) = {v:.4f}")

    # Features on test
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

    # Bootstrap CI on t=final head AUC
    with torch.no_grad():
        probs_final, ys_final = [], []
        for x, y in test_ld:
            x = x.to(DEVICE)
            _, cum = model(x)
            probs_final.append(torch.sigmoid(cum[T-1]).cpu().numpy())
            ys_final.append(y.numpy())
    probs_final = np.concatenate(probs_final); ys_final = np.concatenate(ys_final)
    lo, med, hi = bootstrap_auc(probs_final, ys_final, n=1000, seed=seed)
    print(f"\nbootstrap test AUC(t={T})  median={med:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]")

    # Probe AUC per timestep
    probe = {}
    for t in [t for t in [1, 4, 8, 12, 16] if t <= T]:
        m, s = cv_probe(features[t-1], labels, n_folds=5, seed=seed)
        probe[t] = (m, s)
        print(f"  probe AUC t={t:2d}  {m:.3f} ± {s:.3f}")

    # Consecutive cosine (per class)
    consec = {"benign": [], "malignant": []}
    for cls_idx, cls_name in enumerate(["benign", "malignant"]):
        mask = labels == cls_idx
        traj = features[:, mask].mean(axis=1)   # (T, D)
        for ti in range(T - 1):
            consec[cls_name].append(cos_sim(traj[ti], traj[ti+1]))

    # NEW: full pairwise cosine (per class)
    pairwise_cos = {}
    for cls_idx, cls_name in enumerate(["benign", "malignant"]):
        mask = labels == cls_idx
        traj = features[:, mask].mean(axis=1)   # (T, D)
        mat = np.zeros((T, T))
        for i in range(T):
            for j in range(T):
                mat[i, j] = cos_sim(traj[i], traj[j])
        pairwise_cos[cls_name] = mat

    # SynOps
    macs = compute_static_macs(model)
    spike_rates = collect_spike_rates(model, test_ld, T)
    per_conv, total_cum, by_layer_cum = compute_synops(spike_rates, macs, T)

    final_betas = snapshot_betas(model)

    # Save everything for this run
    np.savez(os.path.join(out_dir, "analysis.npz"),
             features=features, labels=labels, probe=probe, consec=consec,
             pairwise_cos_benign=pairwise_cos["benign"],
             pairwise_cos_malignant=pairwise_cos["malignant"],
             test_aucs=test_aucs, boot_ci=(lo, med, hi),
             final_betas=final_betas,
             spike_rates=spike_rates, macs=macs,
             synops_per_conv_per_t=per_conv,
             synops_total_cum=total_cum,
             synops_by_layer_cum=by_layer_cum,
             probs_final_test=probs_final, ys_final_test=ys_final)

    print(f"\n✓ artifacts → {out_dir}")

    # Free GPU memory before next config
    del model, opt, sched, best_state
    del features, probs_final
    gc.collect()
    torch.cuda.empty_cache()
    return out_dir

# ╔══════════════════════════════════════════════════════════════════════╗
# ║ MAIN LOOP                                                             ║
# ╚══════════════════════════════════════════════════════════════════════╝
completed = []
failed = []
overall_start = time.time()
for i, cfg in enumerate(RUN_MATRIX, 1):
    print(f"\n\n{'#'*72}\n#  RUN {i}/{len(RUN_MATRIX)}  |  elapsed so far: {(time.time()-overall_start)/60:.1f} min\n{'#'*72}")
    try:
        od = train_one(cfg)
        completed.append((cfg, od))
    except Exception as e:
        print(f"\n!!! FAILED: {cfg} — {type(e).__name__}: {e}")
        failed.append((cfg, str(e)))
        gc.collect()
        torch.cuda.empty_cache()

print(f"\n\n{'='*72}\nTRAINING PHASE DONE — {len(completed)} completed, {len(failed)} failed\n" +
      f"total: {(time.time()-overall_start)/60:.1f} min\n{'='*72}")
for cfg, od in completed:
    print(f"  ✓ {cfg['experiment']:>17s} s{cfg['seed']:>4} T{cfg['T']:>2}  → {od}")
for cfg, err in failed:
    print(f"  ✗ {cfg['experiment']:>17s} s{cfg['seed']:>4} T{cfg['T']:>2}  ({err[:80]})")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║ CONSOLIDATION — cross-run figures + tables                            ║
# ╚══════════════════════════════════════════════════════════════════════╝
print("\n\n" + "="*72)
print(" CONSOLIDATING FIGURES + TABLES")
print("="*72)

def load_run(out_dir):
    with open(os.path.join(out_dir, "sidecar.json")) as f:
        cfg = json.load(f)
    d = np.load(os.path.join(out_dir, "analysis.npz"), allow_pickle=True)
    hist = pd.read_csv(os.path.join(out_dir, "history.csv"))
    beta_hist = pd.read_csv(os.path.join(out_dir, "beta_history.csv"))
    return {"cfg": cfg, "npz": d, "history": hist, "beta_history": beta_hist,
            "label": f"{cfg['experiment']}_s{cfg['seed']}_T{cfg['T']}"}

runs = [load_run(od) for _, od in completed]
if not runs:
    print("no completed runs to consolidate — aborting figure phase")
else:
    COLORS = {"baseline_learn": "tab:blue", "matches": "tab:orange", "baseline_frozen": "tab:green"}
    MARKERS = {42: "o", 123: "s", 2024: "^"}

    def label_style(r):
        cfg = r["cfg"]
        if cfg["T"] == 4:
            return dict(color="tab:red", marker="D", linestyle="--", label=f"baseline_learn s42 T=4")
        return dict(color=COLORS.get(cfg["experiment"], "gray"),
                    marker=MARKERS.get(cfg["seed"], "x"), linestyle="-",
                    label=f"{cfg['experiment']} s{cfg['seed']}")

    # ── Figure 4: consecutive cosine (all runs, one line each, benign+malignant averaged) ──
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in runs:
        T = r["cfg"]["T"]
        cb = r["npz"]["consec"].item()["benign"]
        cm = r["npz"]["consec"].item()["malignant"]
        avg = [(b+m)/2 for b,m in zip(cb, cm)]
        st = label_style(r)
        ax.plot(range(1, T), avg, **st, alpha=0.75, linewidth=1.5, markersize=6)
    ax.axhline(0.998, color="gray", linestyle=":", alpha=0.5, label="saturation ≈ 0.998")
    ax.set_xlabel("timestep transition t → t+1")
    ax.set_ylabel("consecutive-timestep feature cosine")
    ax.set_title("Figure 4 — Temporal collapse: feature cosine saturates by t≈4 across all interventions")
    ax.set_ylim(0.9, 1.001)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, "fig4_temporal_collapse.png"), dpi=200, bbox_inches="tight")
    plt.savefig(os.path.join(ARTIFACTS_DIR, "fig4_temporal_collapse.pdf"), bbox_inches="tight")
    plt.show()

    # ── Figure 5: Probe AUC across timesteps ──
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in runs:
        probe = r["npz"]["probe"].item()
        ts = sorted(probe.keys())
        means = [probe[t][0] for t in ts]
        stds  = [probe[t][1] for t in ts]
        st = label_style(r)
        ax.errorbar(ts, means, yerr=stds, **st, alpha=0.75, capsize=3, markersize=6)
    ax.set_xlabel("timestep")
    ax.set_ylabel("5-fold CV probe AUC (test features)")
    ax.set_title("Figure 5 — Feature-quality probe AUC across timesteps")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, "fig5_probe_auc.png"), dpi=200, bbox_inches="tight")
    plt.savefig(os.path.join(ARTIFACTS_DIR, "fig5_probe_auc.pdf"), bbox_inches="tight")
    plt.show()

    # ── Figure 6: Test AUC bar chart with error bars (seed std) ──
    # Group by (experiment, T)
    from collections import defaultdict
    groups = defaultdict(list)
    for r in runs:
        exp = r["cfg"]["experiment"]; T = r["cfg"]["T"]
        # For T=16 use test AUC at t=16; for T=4 use test AUC at t=4
        aucs = r["npz"]["test_aucs"].item()
        t_use = T
        groups[(exp, T)].append(aucs[t_use])
    labels_bar = [f"{e}\n(T={T})" for (e,T) in groups.keys()]
    means = [np.mean(v) for v in groups.values()]
    stds  = [np.std(v)  for v in groups.values()]
    ns    = [len(v)     for v in groups.values()]

    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.arange(len(labels_bar))
    bars = ax.bar(xs, means, yerr=stds, capsize=6, alpha=0.8,
                  color=["tab:blue","tab:orange","tab:green","tab:red"][:len(labels_bar)])
    for i, (m, s, n) in enumerate(zip(means, stds, ns)):
        ax.text(xs[i], m + s + 0.008, f"n={n}\n{m:.3f}±{s:.3f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels(labels_bar)
    ax.set_ylabel("Test AUC (t = T)")
    ax.set_title("Figure 6 — Test AUC by configuration (mean ± seed std)")
    ax.set_ylim(0.7, 0.85)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, "fig6_test_auc_bar.png"), dpi=200, bbox_inches="tight")
    plt.savefig(os.path.join(ARTIFACTS_DIR, "fig6_test_auc_bar.pdf"), bbox_inches="tight")
    plt.show()

    # ── Figure 7: β evolution (baseline_learn, 3 seeds averaged, per layer) ──
    bl_runs = [r for r in runs if r["cfg"]["experiment"] == "baseline_learn" and r["cfg"]["T"] == 16]
    if bl_runs:
        # Aggregate: for each layer, average β across seeds at each logged epoch
        layer_cols = [c for c in bl_runs[0]["beta_history"].columns if c.endswith(".beta")]
        fig, ax = plt.subplots(figsize=(11, 6))
        for lc in layer_cols:
            # concat across seeds; group by epoch, average
            dfs = []
            for r in bl_runs:
                bh = r["beta_history"][["epoch", lc]].copy()
                bh.columns = ["epoch", "beta"]
                dfs.append(bh)
            all_bh = pd.concat(dfs)
            avg = all_bh.groupby("epoch")["beta"].mean().reset_index()
            ax.plot(avg["epoch"], avg["beta"], marker="o", markersize=3,
                    linewidth=1.4, label=lc.replace(".beta", ""))
        ax.set_xlabel("epoch")
        ax.set_ylabel("mean β (learn_beta=True, avg over 3 seeds)")
        ax.set_title("Figure 7 — β evolution during training (baseline_learn, uniform init β=0.95)")
        ax.axhline(0.95, color="gray", linestyle=":", alpha=0.5, label="init β=0.95")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2, loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(ARTIFACTS_DIR, "fig7_beta_evolution.png"), dpi=200, bbox_inches="tight")
        plt.savefig(os.path.join(ARTIFACTS_DIR, "fig7_beta_evolution.pdf"), bbox_inches="tight")
        plt.show()

    # ── Figure 8: Seed consistency scatter (test AUC per seed per config) ──
    fig, ax = plt.subplots(figsize=(9, 5))
    exp_groups = defaultdict(list)
    for r in runs:
        exp = r["cfg"]["experiment"]; T = r["cfg"]["T"]; seed = r["cfg"]["seed"]
        aucs = r["npz"]["test_aucs"].item()
        exp_groups[(exp, T)].append((seed, aucs[T]))
    xs, ys_pts, colors_pts, labels_pts = [], [], [], []
    for i, ((exp, T), seed_auc) in enumerate(exp_groups.items()):
        for (s, a) in seed_auc:
            xs.append(i + (s % 3) * 0.06 - 0.06)   # tiny jitter by seed
            ys_pts.append(a)
    # Repaint properly
    xs_all, ys_all, cs_all = [], [], []
    group_x = {k: i for i, k in enumerate(exp_groups.keys())}
    for r in runs:
        k = (r["cfg"]["experiment"], r["cfg"]["T"])
        aucs = r["npz"]["test_aucs"].item()
        xs_all.append(group_x[k])
        ys_all.append(aucs[r["cfg"]["T"]])
        cs_all.append(COLORS.get(r["cfg"]["experiment"], "tab:red") if r["cfg"]["T"] == 16 else "tab:red")
    ax.scatter(xs_all, ys_all, c=cs_all, s=80, alpha=0.75, edgecolor="black", linewidth=0.5)
    for i, r in enumerate(runs):
        k = (r["cfg"]["experiment"], r["cfg"]["T"])
        ax.annotate(f"s{r['cfg']['seed']}", (group_x[k], ys_all[i]),
                    xytext=(6, 0), textcoords="offset points", fontsize=7)
    ax.set_xticks(list(group_x.values()))
    ax.set_xticklabels([f"{e}\n(T={T})" for (e,T) in group_x.keys()])
    ax.set_ylabel("Test AUC")
    ax.set_title("Figure 8 — Per-seed test AUC (seed-level variance)")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, "fig8_seed_consistency.png"), dpi=200, bbox_inches="tight")
    plt.savefig(os.path.join(ARTIFACTS_DIR, "fig8_seed_consistency.pdf"), bbox_inches="tight")
    plt.show()

    # ── Figure 9: PCA of test features at t=1, t=4, t=16 (baseline_learn s42 T=16) ──
    focus_run = next((r for r in runs
                      if r["cfg"]["experiment"] == "baseline_learn"
                      and r["cfg"]["seed"] == 42 and r["cfg"]["T"] == 16), None)
    if focus_run:
        feats = focus_run["npz"]["features"]   # (T, N, D)
        labs  = focus_run["npz"]["labels"]
        ts_show = [t for t in [1, 4, 16] if t <= feats.shape[0]]
        fig, axes = plt.subplots(1, len(ts_show), figsize=(4*len(ts_show)+1, 4.5))
        if len(ts_show) == 1: axes = [axes]
        for ax_, t in zip(axes, ts_show):
            X = feats[t-1]
            pca = PCA(n_components=2, random_state=42)
            X2 = pca.fit_transform(X)
            for cls_idx, cls_name, col in [(0, "benign", "tab:blue"), (1, "malignant", "tab:red")]:
                mask = labs == cls_idx
                ax_.scatter(X2[mask, 0], X2[mask, 1], c=col, s=25, alpha=0.7, label=cls_name, edgecolor="w", linewidth=0.4)
            ax_.set_title(f"t = {t}")
            ax_.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}%)")
            ax_.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.0f}%)")
            ax_.grid(alpha=0.3)
        axes[-1].legend(loc="best", fontsize=8)
        plt.suptitle(f"Figure 9 — PCA of test features (baseline_learn s42 T=16)", y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(ARTIFACTS_DIR, "fig9_pca_features.png"), dpi=200, bbox_inches="tight")
        plt.savefig(os.path.join(ARTIFACTS_DIR, "fig9_pca_features.pdf"), bbox_inches="tight")
        plt.show()

    # ── Figure 10: Pairwise cosine heatmap (one per T=16 configuration, class-avg) ──
    t16_configs_seen = set()
    heatmap_runs = []
    for r in runs:
        if r["cfg"]["T"] != 16: continue
        key = r["cfg"]["experiment"]
        if key in t16_configs_seen: continue
        t16_configs_seen.add(key)
        heatmap_runs.append(r)
    if heatmap_runs:
        fig, axes = plt.subplots(1, len(heatmap_runs), figsize=(5*len(heatmap_runs), 4.5))
        if len(heatmap_runs) == 1: axes = [axes]
        for ax_, r in zip(axes, heatmap_runs):
            pb = r["npz"]["pairwise_cos_benign"]
            pm = r["npz"]["pairwise_cos_malignant"]
            avg = (pb + pm) / 2
            im = ax_.imshow(avg, cmap="viridis", vmin=0.85, vmax=1.0, origin="lower")
            ax_.set_title(f"{r['cfg']['experiment']}  s{r['cfg']['seed']}")
            ax_.set_xlabel("timestep i")
            ax_.set_ylabel("timestep j")
            ax_.set_xticks([0, 4, 8, 12, 15])
            ax_.set_yticks([0, 4, 8, 12, 15])
            plt.colorbar(im, ax=ax_, fraction=0.046, pad=0.04)
        plt.suptitle("Figure 10 — Pairwise feature cosine (class-averaged, T=16)", y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(ARTIFACTS_DIR, "fig10_pairwise_cosine.png"), dpi=200, bbox_inches="tight")
        plt.savefig(os.path.join(ARTIFACTS_DIR, "fig10_pairwise_cosine.pdf"), bbox_inches="tight")
        plt.show()

    # ── Figure 11: Intervention summary heatmap (rows: config, cols: metrics) ──
    summary_rows = []
    for r in runs:
        cfg = r["cfg"]
        aucs = r["npz"]["test_aucs"].item()
        probe = r["npz"]["probe"].item()
        cb = r["npz"]["consec"].item()["benign"]
        cm = r["npz"]["consec"].item()["malignant"]
        late_cos = float(np.mean(cb[4:] + cm[4:])) if cfg["T"] > 5 else float("nan")
        summary_rows.append({
            "config": f"{cfg['experiment']}_s{cfg['seed']}_T{cfg['T']}",
            "test_auc": aucs[cfg["T"]],
            "probe_final": probe[max(probe.keys())][0],
            "late_cos": late_cos,
            "delta_probe_t4_to_T": (probe[max(probe.keys())][0] - probe[4][0]) if 4 in probe and max(probe.keys()) > 4 else 0.0,
        })
    df_sum = pd.DataFrame(summary_rows).set_index("config")
    fig, ax = plt.subplots(figsize=(9, max(3, 0.5*len(df_sum))))
    im = ax.imshow(df_sum.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(df_sum.columns))); ax.set_xticklabels(df_sum.columns, rotation=15, ha="right")
    ax.set_yticks(range(len(df_sum.index))); ax.set_yticklabels(df_sum.index)
    for i in range(df_sum.shape[0]):
        for j in range(df_sum.shape[1]):
            ax.text(j, i, f"{df_sum.values[i,j]:.3f}", ha="center", va="center", fontsize=8, color="black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Figure 11 — Intervention summary (per-run)")
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, "fig11_intervention_summary.png"), dpi=200, bbox_inches="tight")
    plt.savefig(os.path.join(ARTIFACTS_DIR, "fig11_intervention_summary.pdf"), bbox_inches="tight")
    plt.show()

    # ── Tables ──
    # Table 5 — Main quantitative results (per-seed rows)
    table5 = []
    for r in runs:
        cfg = r["cfg"]
        aucs = r["npz"]["test_aucs"].item()
        probe = r["npz"]["probe"].item()
        lo, med, hi = r["npz"]["boot_ci"]
        row = {
            "experiment": cfg["experiment"], "seed": cfg["seed"], "T": cfg["T"],
            "test_auc_final": aucs[cfg["T"]],
            "boot_ci_lo": float(lo), "boot_ci_hi": float(hi),
            "probe_final_mean": probe[max(probe.keys())][0],
            "probe_final_std":  probe[max(probe.keys())][1],
        }
        table5.append(row)
    pd.DataFrame(table5).to_csv(os.path.join(ARTIFACTS_DIR, "table5_main_results.csv"), index=False)

    # Table 6 — Per-seed comparison (paired matches − baseline_learn at each seed)
    table6 = []
    bl = {r["cfg"]["seed"]: r for r in runs if r["cfg"]["experiment"] == "baseline_learn" and r["cfg"]["T"] == 16}
    mt = {r["cfg"]["seed"]: r for r in runs if r["cfg"]["experiment"] == "matches"        and r["cfg"]["T"] == 16}
    for s in sorted(set(bl.keys()) & set(mt.keys())):
        b_auc = bl[s]["npz"]["test_aucs"].item()[16]
        m_auc = mt[s]["npz"]["test_aucs"].item()[16]
        b_probe = bl[s]["npz"]["probe"].item()[16][0]
        m_probe = mt[s]["npz"]["probe"].item()[16][0]
        table6.append({
            "seed": s,
            "baseline_test": b_auc, "matches_test": m_auc, "delta_test": m_auc - b_auc,
            "baseline_probe": b_probe, "matches_probe": m_probe, "delta_probe": m_probe - b_probe,
        })
    pd.DataFrame(table6).to_csv(os.path.join(ARTIFACTS_DIR, "table6_paired_by_seed.csv"), index=False)

    # Table 7 — Statistical tests
    def paired_permutation(a, b, n=10000, seed=0):
        a = np.asarray(a); b = np.asarray(b)
        obs = np.mean(a - b)
        rng = np.random.default_rng(seed)
        cnt = 0
        for _ in range(n):
            signs = rng.choice([-1, 1], size=len(a))
            perm_mean = np.mean(signs * (a - b))
            if abs(perm_mean) >= abs(obs):
                cnt += 1
        return obs, cnt / n
    def cohens_d_paired(a, b):
        d = np.asarray(a) - np.asarray(b)
        return float(np.mean(d) / (np.std(d, ddof=1) + 1e-12))
    stat_rows = []
    if table6:
        a_test  = [r["matches_test"]  for r in table6]
        b_test  = [r["baseline_test"] for r in table6]
        a_probe = [r["matches_probe"]  for r in table6]
        b_probe = [r["baseline_probe"] for r in table6]
        obs_t, p_t = paired_permutation(a_test, b_test)
        obs_p, p_p = paired_permutation(a_probe, b_probe)
        stat_rows.append({"comparison": "matches − baseline_learn  test AUC",
                          "n_pairs": len(a_test), "mean_diff": obs_t,
                          "paired_perm_p": p_t, "cohens_d_paired": cohens_d_paired(a_test, b_test)})
        stat_rows.append({"comparison": "matches − baseline_learn  probe AUC",
                          "n_pairs": len(a_probe), "mean_diff": obs_p,
                          "paired_perm_p": p_p, "cohens_d_paired": cohens_d_paired(a_probe, b_probe)})
    pd.DataFrame(stat_rows).to_csv(os.path.join(ARTIFACTS_DIR, "table7_statistical_tests.csv"), index=False)

    # Table 8 — Ablation summary (SynOps + AUC per config)
    table8 = []
    for r in runs:
        cfg = r["cfg"]
        aucs = r["npz"]["test_aucs"].item()
        total_cum = r["npz"]["synops_total_cum"]
        row = {"config": f"{cfg['experiment']}_s{cfg['seed']}_T{cfg['T']}"}
        for t in [1, 4, 8, 12, 16]:
            if t <= cfg["T"]:
                row[f"synops_t{t}"] = float(total_cum[t-1])
                row[f"auc_t{t}"] = aucs.get(t, float("nan"))
        if cfg["T"] == 16:
            row["synops_ratio_t16_t4"] = float(total_cum[15] / total_cum[3])
            row["auc_gain_t4_to_t16"]  = aucs[16] - aucs[4]
        table8.append(row)
    pd.DataFrame(table8).to_csv(os.path.join(ARTIFACTS_DIR, "table8_ablation_synops.csv"), index=False)

    print("\nTables written:")
    for f in sorted(os.listdir(ARTIFACTS_DIR)):
        if f.endswith(".csv"):
            print(f"  {f}")

    print("\nFigures written:")
    for f in sorted(os.listdir(ARTIFACTS_DIR)):
        if f.endswith((".png", ".pdf")):
            print(f"  {f}")

    # Compact summary print
    print("\n" + "="*72)
    print(" QUICK SUMMARY (test AUCs)")
    print("="*72)
    df5 = pd.DataFrame(table5)
    print(df5.to_string(index=False))

print(f"\n\nALL DONE. total wall time: {(time.time()-overall_start)/60:.1f} min")
print(f"All outputs → {WORK_ROOT}/")