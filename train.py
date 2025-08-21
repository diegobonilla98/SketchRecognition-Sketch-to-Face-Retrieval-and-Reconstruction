import os
import sys
import math
import time
import random
import glob
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

import onnx
from onnx2torch import convert

import insightface
from insightface.utils import face_align as fa

import open_clip
from copy import deepcopy

import torchvision.transforms as T

# ---------------------------
# CONFIG
# ---------------------------
@dataclass
class Config:
    dataset_path: str = r"F:\FaceSketch\full"
    arcface_onnx: str = r"D:\hf\hub\w600k_r50.onnx"
    out_dir: str = r"./runs/sketch2arcfacev2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    train_ratio: float = 0.95
    seed: int = 42
    num_workers: int = 6
    pin_memory: bool = True

    # Model
    clip_model: str = "ViT-B-16"
    clip_pretrained: str = "openai"
    proj_hidden: int = 1024
    embed_dim: int = 512

    # Training
    batch_size: int = 64  # 384
    epochs: int = 50
    fp16: bool = True
    grad_clip_norm: float = 1.0
    accum_steps: int = 6 # 1

    # Optim
    lr_head: float = 1e-3
    lr_backbone: float = 3e-5
    wd: float = 1e-2
    warmup_steps: int = 1000
    cosine_final_lr_mul: float = 0.05

    # ArcFace (true angular margin)
    scale_s: float = 64.0
    margin_m: float = 0.30
    sampled_softmax_classes: int = 8192   # <= total IDs; includes all positives + sampled negatives

    # Regularizers
    lambda_consistency: float = 0.1
    lambda_coral: float = 0.005
    lambda_reg: float = 1.0              # direct regression to its class prototype

    # Eval negatives per query (sampled)
    val_negatives_per_query: int = 200

    # Gradual unfreezing
    unfreeze_after_steps: int = 2000
    unfreeze_last_blocks: int = 4

    # Semi-hard negatives (optional)
    use_hard_negs: bool = True
    hard_frac: float = 0.5
    hard_topk: int = 4096

    # Checkpointing / logging
    log_every: int = 1
    eval_every_epochs: int = 1
    save_best: bool = True
    resume: Optional[str] = None  # path to checkpoint to resume from (optional)

CFG = Config()

# ---------------------------
# UTILITIES
# ---------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(p=2, dim=1, keepdim=True).clamp_min(eps)

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(-1)

# ---------------------------
# ARC FACE WRAPPER (teacher)
# ---------------------------
class ArcFaceW600kR50(nn.Module):
    """
    PyTorch wrapper around the exact buffalo_l ArcFace model (w600k_r50.onnx).
    Forward returns L2-normalized 512-D embeddings.
    """
    def __init__(self, onnx_path: str, device: str = "cuda"):
        super().__init__()
        model_onnx = onnx.load(onnx_path)
        self.device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        self.core = convert(model_onnx).to(self.device)
        self.core.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,3,112,112) float32 in [-1,1]
        out = self.core(x.to(self.device))
        if isinstance(out, (list, tuple)):
            out = out[0]
        out = out.float()
        return l2_normalize(out)

def preprocess_for_arcface(img: Image.Image) -> torch.Tensor:
    """
    Expect a PIL RGB image. Resize to 112x112 and map to [-1, 1].
    Return FloatTensor of shape (1,3,112,112).
    """
    img = img.convert("RGB").resize((112, 112), Image.BILINEAR)
    x = np.asarray(img).astype(np.float32)  # HWC, RGB, [0..255]
    x = (x - 127.5) / 127.5                 # -> [-1, 1]
    x = np.transpose(x, (2, 0, 1))          # CHW
    return torch.from_numpy(x).unsqueeze(0) # 1x3x112x112

# ---------------------------
# FACE ALIGNMENT (photos only)
# ---------------------------
def align_photo(img: Image.Image) -> Image.Image:
    """
    Aligns face using InsightFace 5-pt landmarks to ArcFace's 112x112 template.
    Falls back to a centered crop if alignment fails or insightface not available.
    """
    try:
        if not hasattr(align_photo, "_fa"):
            align_photo._fa = insightface.app.FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider","CPUExecutionProvider"])
            align_photo._fa.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))
        fa_app = align_photo._fa

        img_np = np.array(img.convert("RGB"))
        faces = fa_app.get(img_np)
        if len(faces) == 0:
            w, h = img.size
            side = int(0.8 * min(w, h))
            img_c = ImageOps.fit(img.convert("RGB"), (side, side), Image.BILINEAR, centering=(0.5, 0.45))
            return img_c.resize((112, 112), Image.BILINEAR)
        # choose largest face
        faces.sort(key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
        kp = faces[0].kps  # (5,2) numpy
        aligned = fa.norm_crop(img_np, landmark=kp, image_size=112)
        return Image.fromarray(aligned)
    except Exception as e:
        print(f"[warn] alignment failed, fallback center crop: {e}", file=sys.stderr)
        w, h = img.size
        side = int(0.8 * min(w, h))
        img_c = ImageOps.fit(img.convert("RGB"), (side, side), Image.BILINEAR, centering=(0.5, 0.45))
        return img_c.resize((112, 112), Image.BILINEAR)

# ---------------------------
# DATASET
# ---------------------------
def build_pairs(dataset_path: str) -> List[Tuple[str, str]]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.webp")
    real_files = []
    for ext in exts:
        real_files += glob.glob(os.path.join(dataset_path, "real", ext))
    pairs = []
    for real_path in real_files:
        stem = os.path.splitext(os.path.basename(real_path))[0]
        sketch_path = None
        for ext in ("png","jpg","jpeg","webp"):
            cand = os.path.join(dataset_path, "sketch", f"{stem}.{ext}")
            if os.path.exists(cand):
                sketch_path = cand
                break
        if sketch_path is not None:
            pairs.append((real_path, sketch_path))
    if len(pairs) == 0:
        raise RuntimeError("No pairs found. Expected dataset_path/real/<id>.(png|jpg|jpeg|webp) with matching sketch/<id>.*")
    pairs.sort()
    return pairs

class SketchFaceDataset(Dataset):
    """
    Operates on pairs:
      real/<id>.*  <->  sketch/<id>.*
    Identity id is filename stem.
    """
    def __init__(self, pairs: List[Tuple[str, str]], id_to_index: Dict[str, int],
                 sketch_transform, sketch_transform_strong, return_two_views: bool = True):
        self.pairs = pairs
        self.id_to_index = id_to_index
        self.sketch_transform = sketch_transform
        self.sketch_transform_strong = sketch_transform_strong
        self.return_two_views = return_two_views

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        real_path, sketch_path = self.pairs[idx]
        identity_id = os.path.splitext(os.path.basename(real_path))[0]
        y = self.id_to_index[identity_id]

        sketch = Image.open(sketch_path).convert("RGB")
        if self.return_two_views:
            v1 = self.sketch_transform(sketch)
            v2 = self.sketch_transform_strong(sketch)
            return v1, v2, y
        else:
            v = self.sketch_transform(sketch)
            return v, y

# ---------------------------
# CLIP BACKBONE + HEAD
# ---------------------------
class SketchEncoder(nn.Module):
    def __init__(self, clip_model="ViT-B-16", pretrained="openai", proj_hidden=1024, out_dim=512, freeze_backbone=True):
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms(clip_model, pretrained=pretrained)
        self.clip = model.visual  # image tower only
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.clip.parameters():
                p.requires_grad = False

        # Discover CLIP visual output dim robustly
        clip_width = getattr(self.clip, "output_dim", None)
        if clip_width is None:
            with torch.no_grad():
                dev = next(self.clip.parameters()).device
                dummy = torch.zeros(1, 3, 224, 224, device=dev)
                clip_width = self.clip(dummy).shape[-1]

        self.head = nn.Sequential(
            nn.Linear(clip_width, proj_hidden),
            nn.LayerNorm(proj_hidden),
            nn.GELU(),
            nn.Linear(proj_hidden, out_dim)
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.clip(x)                 # (N, clip_width)
        z = self.head(feats)                 # (N, out_dim)
        z = l2_normalize(z)
        return z

    def unfreeze_last_blocks(self, n_blocks: int = 2):
        """
        Gradually unfreeze last N transformer blocks for open_clip ViTs.
        """
        # ViT path
        if hasattr(self.clip, "trunk") and hasattr(self.clip.trunk, "blocks"):
            blocks = self.clip.trunk.blocks
            for blk in blocks[-n_blocks:]:
                for p in blk.parameters():
                    p.requires_grad = True
        elif hasattr(self.clip, "transformer") and hasattr(self.clip.transformer, "resblocks"):
            blocks = self.clip.transformer.resblocks
            for blk in blocks[-n_blocks:]:
                for p in blk.parameters():
                    p.requires_grad = True
        else:
            # Fallback: unfreeze last params encountered (crude)
            n = n_blocks
            for _, module in reversed(list(self.clip.named_modules())):
                if n <= 0:
                    break
                if isinstance(module, (nn.Linear, nn.Conv2d, nn.TransformerEncoderLayer)):
                    for p in module.parameters():
                        p.requires_grad = True
                    n -= 1

# ---------------------------
# AUGMENTATIONS (sketch)
# ---------------------------
def build_sketch_transforms(image_size=224):
    # CLIP mean/std
    mean = (0.48145466, 0.4578275, 0.40821073)
    std  = (0.26862954, 0.26130258, 0.27577711)

    weak = T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomAffine(degrees=12, translate=(0.1,0.1), scale=(0.9,1.1), shear=None, interpolation=T.InterpolationMode.BICUBIC),
        T.RandomPerspective(distortion_scale=0.08, p=0.3, interpolation=T.InterpolationMode.BICUBIC),
        T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.3),
        T.RandomApply([T.RandomAdjustSharpness(sharpness_factor=1.5)], p=0.3),
        T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.0, hue=0.0)], p=0.5),  # no hue/sat
        T.RandomInvert(p=0.2),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    strong = T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.85, 1.15), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomAffine(degrees=18, translate=(0.15,0.15), scale=(0.85,1.15), shear=(-5,5), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomPerspective(distortion_scale=0.12, p=0.5, interpolation=T.InterpolationMode.BICUBIC),
        T.RandomApply([T.GaussianBlur(kernel_size=5)], p=0.4),
        T.RandomApply([T.RandomAdjustSharpness(sharpness_factor=2.0)], p=0.4),
        T.RandomApply([T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.0, hue=0.0)], p=0.7),
        T.RandomInvert(p=0.3),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    eval_tx = T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return weak, strong, eval_tx

# ---------------------------
# PROTOTYPES (photo -> ArcFace)
# ---------------------------
def compute_prototypes(pairs: List[Tuple[str,str]], model_arc: ArcFaceW600kR50) -> Tuple[torch.Tensor, Dict[str,int], List[str]]:
    """
    Returns:
      W: (C,512) tensor of normalized photo prototypes (fixed, on model's device)
      id_to_index: dict id -> idx
      index_to_id: list of ids by row index
    """
    id_to_index: Dict[str,int] = {}
    index_to_id: List[str] = []
    embs = []

    print("[info] Precomputing ArcFace photo prototypes...")
    for real_path, _ in tqdm(pairs, ncols=100):
        identity_id = os.path.splitext(os.path.basename(real_path))[0]
        id_to_index[identity_id] = len(index_to_id)
        index_to_id.append(identity_id)

        img = Image.open(real_path).convert("RGB")
        img_aligned = align_photo(img)
        x = preprocess_for_arcface(img_aligned).to(model_arc.device)  # (1,3,112,112)
        with torch.no_grad():
            emb = model_arc(x).cpu().numpy().astype(np.float32).squeeze(0)  # (512,)
        embs.append(emb)

    W = torch.from_numpy(np.stack(embs, axis=0)).to(model_arc.device)  # (C,512), L2-normalized
    return W, id_to_index, index_to_id

def load_or_compute_prototypes(dataset_root: str, pairs: List[Tuple[str, str]],
                               model_arc: ArcFaceW600kR50) -> Tuple[torch.Tensor, Dict[str,int], List[str]]:
    """
    Load cached ArcFace photo prototypes if available and consistent with dataset;
    otherwise compute and save to `<dataset_root>/arcface_prototypes.npz`.
    """
    cache_file = os.path.join(dataset_root, "arcface_prototypes.npz")
    ordered_ids = [os.path.splitext(os.path.basename(r))[0] for r, _ in pairs]
    ordered_ids_list = list(ordered_ids)

    if os.path.isfile(cache_file):
        try:
            data = np.load(cache_file, allow_pickle=True)
            cache_ids = list(data["ids"].tolist())
            W_np = data["W"].astype(np.float32)
            if set(cache_ids) == set(ordered_ids_list) and W_np.shape[0] == len(cache_ids) and W_np.shape[1] == 512:
                # Reorder rows to match current ordered_ids
                row_by_id = {i: idx for idx, i in enumerate(cache_ids)}
                idx_order = np.array([row_by_id[i] for i in ordered_ids_list], dtype=np.int64)
                W_np = W_np[idx_order]
                W = torch.from_numpy(W_np).to(model_arc.device)
                id_to_index = {i: k for k, i in enumerate(ordered_ids_list)}
                print(f"[info] Loaded ArcFace prototypes cache: {cache_file}")
                return W, id_to_index, ordered_ids_list
            else:
                print(f"[warn] Prototype cache mismatch with dataset; recomputing ({cache_file}).")
        except Exception as e:
            print(f"[warn] Failed to load prototype cache ({cache_file}): {e}; recomputing.")

    # Compute and cache
    W, id_to_index, index_to_id = compute_prototypes(pairs, model_arc)
    try:
        np.savez(cache_file,
                 ids=np.array(index_to_id, dtype=object),
                 W=W.detach().cpu().numpy().astype(np.float32))
        print(f"[info] Saved ArcFace prototypes cache to: {cache_file}")
    except Exception as e:
        print(f"[warn] Could not save prototype cache to {cache_file}: {e}")
    return W, id_to_index, index_to_id

# ---------------------------
# LOSSES (ArcFace angular margin with sampled proxies)
# ---------------------------
def arcface_sampled_softmax(z: torch.Tensor, labels: torch.Tensor,
                            W_full: torch.Tensor, s: float, m: float,
                            sampled_classes: int, rng: torch.Generator,
                            use_hard_negs: bool = True, hard_frac: float = 0.5, hard_topk: int = 4096) -> Tuple[torch.Tensor, int]:
    """
    ArcFace sampled softmax with optional semi-hard negatives.
    """
    device = z.device
    B, C = z.size(0), W_full.size(0)

    # ensure we include all positives
    pos_classes = labels.unique()
    n_pos = pos_classes.numel()

    Cs_target = min(sampled_classes, C)
    k_needed = max(Cs_target - n_pos, 0)

    hard_idx = torch.empty(0, dtype=torch.long, device=device)
    if use_hard_negs and k_needed > 0:
        sims = (z @ W_full.T).to(torch.float32)
        sims[torch.arange(B, device=device), labels] = -1.0
        topk = min(hard_topk, C - 1)
        hard_per_sample = sims.topk(k=topk, dim=1, largest=True, sorted=False).indices
        hard_idx = torch.unique(hard_per_sample.reshape(-1))
        hard_mask = torch.ones(C, dtype=torch.bool, device=device)
        hard_mask[pos_classes] = False
        hard_idx = hard_idx[hard_mask[hard_idx]]

    k_hard = min(int(k_needed * hard_frac), hard_idx.numel())
    chosen_hard = hard_idx[:k_hard] if k_hard > 0 else torch.empty(0, dtype=torch.long, device=device)

    k_rand = k_needed - k_hard
    if k_rand > 0:
        all_idx = torch.arange(C, device=device)
        mask = torch.ones(C, dtype=torch.bool, device=device)
        mask[pos_classes] = False
        if k_hard > 0:
            mask[chosen_hard] = False
        pool = all_idx[mask]
        perm = torch.randperm(pool.numel(), generator=rng, device=device)
        rand_idx = pool[perm[:k_rand]]
        neg_idx = torch.cat([chosen_hard, rand_idx], dim=0)
    else:
        neg_idx = chosen_hard

    cls_idx = torch.cat([pos_classes, neg_idx], dim=0)
    W = W_full[cls_idx]  # (Cs,512)

    # local label mapping on GPU (vectorized)
    eq = (cls_idx.view(1, -1) == labels.view(-1, 1))
    assert torch.all(eq.any(dim=1)), "Label missing from sampled classes."
    local_labels = eq.float().argmax(dim=1)

    # Keep tensor-core matmul speed, then upcast for stable margin math
    cos = (z @ W.T).to(torch.float32).clamp(-1+1e-7, 1-1e-7)
    sin = torch.sqrt((1.0 - cos**2).clamp_min(0.0))
    phi = cos * math.cos(m) - sin * math.sin(m)   # cos(θ + m)

    logits = s * cos
    rows = torch.arange(B, device=device)
    logits[rows, local_labels] = (s * phi)[rows, local_labels]

    loss = F.cross_entropy(logits, local_labels.long())
    return loss, W.size(0)

def coral_loss(z_batch: torch.Tensor, proto_sample: torch.Tensor) -> torch.Tensor:
    """
    Simple CORAL: match mean and covariance between z_batch and proto_sample.
    Both expected L2-normalized. Shapes: (B,512) and (K,512)
    """
    def _moments(x):
        mu = x.mean(dim=0, keepdim=True)
        xc = x - mu
        cov = (xc.T @ xc) / (x.size(0) - 1 + 1e-6)
        return mu, cov

    mu_s, cov_s = _moments(z_batch.float())
    mu_p, cov_p = _moments(proto_sample.float())
    return (mu_s - mu_p).pow(2).sum() + (cov_s - cov_p).pow(2).sum()

# ---------------------------
# SCHEDULES
# ---------------------------
def sm_schedule(opt_step):
    t = opt_step / max(1, CFG.warmup_steps * 2)
    ramp = min(1.0, max(0.0, t))
    s_t = 16.0 + (CFG.scale_s - 16.0) * ramp
    m_t = 0.0  + (CFG.margin_m - 0.0) * ramp
    return s_t, m_t

def update_ema(student: nn.Module, teacher: nn.Module, decay: float):
    with torch.no_grad():
        sdict = student.state_dict()
        for k, v in teacher.state_dict().items():
            v.copy_(v * decay + sdict[k] * (1.0 - decay))

def ema_decay_schedule(opt_step: int, steps_per_epoch: int) -> float:
    # ramp 0.996 -> 0.999 over ~10 epochs
    t = min(1.0, opt_step / max(1, 10 * steps_per_epoch))
    return 0.996 + (0.999 - 0.996) * t

def lambda_cons_schedule(opt_step: int, unfreeze_step: int, steps_per_epoch: int, base_lambda: float) -> float:
    if opt_step < unfreeze_step:
        return base_lambda
    # linear decay to 20% over ~3 epochs post-unfreeze
    t = (opt_step - unfreeze_step) / max(1, 3 * steps_per_epoch)
    return base_lambda * max(0.2, 1.0 - t)

# ---------------------------
# EVALUATION
# ---------------------------
@torch.no_grad()
def evaluate(val_loader, model_s: nn.Module, W_full: torch.Tensor, W_val: torch.Tensor,
             val_global_indices: torch.Tensor, cfg: Config, writer: Optional[SummaryWriter], step: int, device: str):
    model_s.eval()
    Z_list, Y_list = [], []
    for batch in tqdm(val_loader, desc="Val", ncols=100, leave=False):
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            v1, v2, y = batch
            x = v1.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
        else:
            x, y = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
        z = model_s(x)  # (B,512)
        Z_list.append(z)
        Y_list.append(y)

    Z = torch.cat(Z_list, dim=0)  # device
    Y = torch.cat(Y_list, dim=0)

    # R@1 with FULL gallery
    sims_full = (Z @ W_full.T)               # (N, Cfull)
    top1_full = sims_full.argmax(dim=1)
    recall1_full = (top1_full == Y).float().mean().item()

    # R@1 with VAL-only gallery
    sims_val = (Z @ W_val.T)                 # (N, Cval)
    top1_local = sims_val.argmax(dim=1)
    # map local idx back to global class index
    top1_global = val_global_indices[top1_local]
    recall1_val = (top1_global == Y).float().mean().item()

    # Verification TPR@FPR with FULL gallery (sampled negatives)
    N = Z.size(0)
    pos = cosine_sim(Z, W_full[Y])  # (N,)
    C = W_full.size(0)
    K = min(cfg.val_negatives_per_query, C-1)
    rng = torch.Generator(device=device)
    rng.manual_seed(0)

    idx_all = torch.arange(C, device=device)
    neg_scores = []
    for i in range(N):
        mask = torch.ones(C, dtype=torch.bool, device=device)
        mask[Y[i]] = False
        pool = idx_all[mask]
        perm = torch.randperm(pool.numel(), generator=rng, device=device)
        neg_idx = pool[perm[:K]]
        neg_scores.append((Z[i:i+1] @ W_full[neg_idx].T).squeeze(0))
    neg = torch.cat(neg_scores, dim=0)

    def tpr_at_fpr(target_fpr: float) -> float:
        if neg.numel() == 0:
            return 0.0
        thr = torch.quantile(neg, 1.0 - target_fpr)
        return (pos >= thr).float().mean().item()

    tpr_1e2 = tpr_at_fpr(1e-2)
    tpr_1e3 = tpr_at_fpr(1e-3)

    if writer is not None:
        writer.add_scalar("val/recall@1_full", recall1_full, step)
        writer.add_scalar("val/recall@1_valgallery", recall1_val, step)
        writer.add_scalar("val/TPR@FPR1e-2", tpr_1e2, step)
        writer.add_scalar("val/TPR@FPR1e-3", tpr_1e3, step)
        writer.add_histogram("val/pos_sim", pos.detach().cpu().numpy(), step)
        writer.add_histogram("val/neg_sim", neg.detach().cpu().numpy(), step)

    model_s.train()
    return recall1_full, recall1_val, tpr_1e2, tpr_1e3

# ---------------------------
# TRAIN
# ---------------------------
def main(cfg: Config):
    os.makedirs(cfg.out_dir, exist_ok=True)
    set_seed(cfg.seed)

    device = torch.device(cfg.device)
    writer = SummaryWriter(log_dir=cfg.out_dir, flush_secs=1)
    # Write an initial ping so TensorBoard immediately sees this run
    writer.add_text("run/status", "initialized", 0)
    writer.add_text("run/logdir", cfg.out_dir, 0)
    writer.flush()
    # If we are on a problematic filesystem (e.g., cloud drive on Windows),
    # event files can stay at header-only size. Detect and fall back to a
    # local temp directory so TensorBoard can read updates live.
    try:
        time.sleep(0.1)
        logdir_abs = os.path.abspath(cfg.out_dir)
        evt_files = []
        for root, _, files in os.walk(logdir_abs):
            for f in files:
                if f.startswith("events.out.tfevents"):
                    evt_files.append(os.path.join(root, f))
        size_max = max((os.path.getsize(p) for p in evt_files), default=0)
        if size_max <= 88:  # header-only -> likely blocked writes
            fallback_base = os.environ.get("LOCALAPPDATA") or os.environ.get("TEMP") or "C:\\temp"
            fallback_dir = os.path.join(fallback_base, "sketch2arcface_tb")
            os.makedirs(fallback_dir, exist_ok=True)
            print(f"[warn] TensorBoard writes look blocked at '{logdir_abs}'. Falling back to '{fallback_dir}'.")
            try:
                writer.close()
            except Exception:
                pass
            writer = SummaryWriter(log_dir=fallback_dir, flush_secs=1)
            writer.add_text("run/status", f"fallback from {logdir_abs}", 0)
            writer.flush()
            print(f"[tb] Now logging to: {os.path.abspath(fallback_dir)}")
    except Exception as _e:
        print(f"[warn] TB fallback check failed: {_e}")

    # 1) Build pairs & IDs
    pairs_all = build_pairs(cfg.dataset_path)
    random.shuffle(pairs_all)

    # 2) Load ArcFace teacher and precompute photo prototypes (with caching)
    arc = ArcFaceW600kR50(cfg.arcface_onnx, device=cfg.device)
    dataset_root = cfg.dataset_path
    W_full, id_to_index, index_to_id = load_or_compute_prototypes(dataset_root, pairs_all, arc)  # on arc.device
    W_full = W_full.to(device)  # ensure same device as training

    # 3) Train/val split over identities
    identities = list(id_to_index.keys())
    random.shuffle(identities)
    n_train = int(len(identities) * cfg.train_ratio)
    train_ids = set(identities[:n_train])
    val_ids = set(identities[n_train:])

    def split_pairs(pairs):
        train, val = [], []
        for real_path, sketch_path in pairs:
            identity_id = os.path.splitext(os.path.basename(real_path))[0]
            if identity_id in train_ids:
                train.append((real_path, sketch_path))
            else:
                val.append((real_path, sketch_path))
        return train, val

    train_pairs, val_pairs = split_pairs(pairs_all)
    print(f"[info] Total IDs: {len(identities)} | Train IDs: {len(train_ids)} | Val IDs: {len(val_ids)}")
    print(f"[info] Train pairs: {len(train_pairs)} | Val pairs: {len(val_pairs)}")

    # Build index tensors for val-only gallery
    val_global_indices = torch.tensor([id_to_index[i] for i in sorted(val_ids)], device=device, dtype=torch.long)
    W_val = W_full.index_select(0, val_global_indices)

    # 4) Sketch transforms
    weak_tx, strong_tx, eval_tx = build_sketch_transforms(image_size=224)

    # 5) Datasets / loaders
    train_ds = SketchFaceDataset(train_pairs, id_to_index, weak_tx, strong_tx, return_two_views=True)
    val_ds   = SketchFaceDataset(val_pairs,   id_to_index, eval_tx, eval_tx, return_two_views=False)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)

    # Make schedules relative to dataset size and accumulation
    steps_per_epoch_micro = len(train_loader)
    steps_per_epoch = max(1, math.ceil(steps_per_epoch_micro / max(1, CFG.accum_steps)))
    CFG.warmup_steps = max(50, min(400, 2 * steps_per_epoch))
    CFG.unfreeze_after_steps = max(steps_per_epoch, 2 * steps_per_epoch)
    total_opt_steps = CFG.epochs * steps_per_epoch
    print(f"[info] accum_steps={CFG.accum_steps} | steps/epoch (micro)={steps_per_epoch_micro} | steps/epoch (opt)={steps_per_epoch}")
    print(f"[info] warmup_steps={CFG.warmup_steps} | unfreeze_after_steps={CFG.unfreeze_after_steps} | total_opt_steps={total_opt_steps}")

    # 6) Student model (sketch encoder)
    model_s = SketchEncoder(cfg.clip_model, cfg.clip_pretrained, cfg.proj_hidden, cfg.embed_dim, freeze_backbone=True).to(device)

    # Mean-Teacher EMA model (teacher)
    model_ema = deepcopy(model_s).to(device).eval()
    for p in model_ema.parameters():
        p.requires_grad = False

    # 7) Optimizer param groups (build from requires_grad to match potential resume state)
    head_params = [p for p in model_s.head.parameters() if p.requires_grad]
    bb_params   = [p for p in model_s.clip.parameters() if p.requires_grad]
    param_groups = [{"params": head_params, "lr": cfg.lr_head}]
    if bb_params:
        param_groups.append({"params": bb_params, "lr": cfg.lr_backbone})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.wd)

    total_steps = cfg.epochs * steps_per_epoch
    def lr_lambda(opt_step_val):
        if opt_step_val < cfg.warmup_steps:
            return (opt_step_val + 1) / max(1, cfg.warmup_steps)
        progress = (opt_step_val - cfg.warmup_steps) / max(1, (total_steps - cfg.warmup_steps))
        return cfg.cosine_final_lr_mul + 0.5 * (1 - cfg.cosine_final_lr_mul) * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.fp16)

    # 8) Resume (optional)
    start_epoch, opt_step, best_recall_val = 0, 0, -1.0
    if cfg.resume and os.path.isfile(cfg.resume):
        ckpt = torch.load(cfg.resume, map_location="cpu")
        model_s.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0)
        opt_step = ckpt.get("global_step", 0)
        best_recall_val = ckpt.get("best_recall_val", -1.0)
        if "model_ema" in ckpt:
            try:
                model_ema.load_state_dict(ckpt["model_ema"])
            except Exception:
                model_ema.load_state_dict(model_s.state_dict())
        else:
            model_ema.load_state_dict(model_s.state_dict())
        # align scheduler epoch to optimizer steps
        scheduler.last_epoch = opt_step
        print(f"[info] Resumed from {cfg.resume} (epoch {start_epoch}, opt_step {opt_step}, best R@1(val) {best_recall_val:.4f})")

    torch.backends.cudnn.benchmark = True
    rng = torch.Generator(device=device)
    rng.manual_seed(cfg.seed)

    # 9) Train
    did_unfreeze = any((p.requires_grad for p in model_s.clip.parameters()))
    for epoch in range(start_epoch, cfg.epochs):
        model_s.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}", ncols=120)
        epoch_loss = epoch_ce = epoch_cons = epoch_coral = epoch_reg = 0.0
        epoch_seen = 0
        Cs_last = 0
        t0 = time.time()

        for it, batch in enumerate(pbar):
            v1, v2, y = batch
            x1 = v1.to(device, non_blocking=True)
            x2 = v2.to(device, non_blocking=True)
            y  = y.to(device, non_blocking=True)

            # Gradual unfreeze (idempotent): add new params, rebuild scheduler to include new group
            if (not did_unfreeze) and cfg.unfreeze_last_blocks > 0 and opt_step >= cfg.unfreeze_after_steps:
                print(f"\n[info] Unfreezing last {cfg.unfreeze_last_blocks} blocks at opt_step {opt_step}")
                model_s.unfreeze_last_blocks(cfg.unfreeze_last_blocks)

                existing = {p for g in optimizer.param_groups for p in g["params"]}
                new_bb_params = [p for p in model_s.clip.parameters() if p.requires_grad and p not in existing]
                if new_bb_params:
                    optimizer.add_param_group({"params": new_bb_params, "lr": cfg.lr_backbone})
                    # Recreate scheduler so it tracks ALL groups, then align to current step
                    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
                    scheduler.last_epoch = opt_step
                    print(f"[info] Added {len(new_bb_params)} backbone params to optimizer and rebuilt scheduler.")
                did_unfreeze = True

            if (it % cfg.accum_steps) == 0:
                optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.fp16):
                z1 = model_s(x1)  # weak view (student)
                z2 = model_s(x2)  # strong view (student)

                # primary classification loss with sampled ArcFace softmax
                s_t, m_t = sm_schedule(opt_step)
                ce1, Cs1 = arcface_sampled_softmax(
                    z1, y, W_full, s_t, m_t,
                    cfg.sampled_softmax_classes, rng,
                    use_hard_negs=cfg.use_hard_negs, hard_frac=cfg.hard_frac, hard_topk=cfg.hard_topk
                )
                ce2, Cs2 = arcface_sampled_softmax(
                    z2, y, W_full, s_t, m_t,
                    cfg.sampled_softmax_classes, rng,
                    use_hard_negs=cfg.use_hard_negs, hard_frac=cfg.hard_frac, hard_topk=cfg.hard_topk
                )
                ce = 0.5 * (ce1 + ce2)
                Cs_last = Cs1

                # Mean-Teacher consistency: teacher on WEAK vs student on STRONG
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=cfg.fp16):
                        zt = model_ema(x1)
                cons = (1.0 - cosine_sim(z2, zt)).mean()

                # direct regression to prototype (use strong view)
                v_t = W_full[y]  # (B,512), float32
                reg = (1.0 - cosine_sim(z2.float(), v_t)).mean()

                # CORAL / distribution alignment (match 2B samples)
                with torch.no_grad():
                    C = W_full.size(0)
                    K = 2 * z1.size(0)
                    idx = torch.randint(low=0, high=C, size=(K,), generator=rng, device=device)
                    proto_sample = W_full[idx]  # (K,512)
                coral = coral_loss(torch.cat([z1, z2], dim=0), proto_sample)

                # Decay consistency weight after unfreeze
                lam_cons = lambda_cons_schedule(opt_step, cfg.unfreeze_after_steps, steps_per_epoch, cfg.lambda_consistency)
                loss_raw = ce + lam_cons * cons + cfg.lambda_coral * coral + cfg.lambda_reg * reg

            # Proper AMP backward + optimizer step with optional grad clipping and accumulation
            scaler.scale(loss_raw / max(1, cfg.accum_steps)).backward()
            do_step = ((it + 1) % cfg.accum_steps == 0) or (it == len(train_loader) - 1)
            if do_step:
                if cfg.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model_s.parameters(), cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1

                # Update EMA teacher
                decay = ema_decay_schedule(opt_step, steps_per_epoch)
                update_ema(model_s, model_ema, decay)

            bs = y.size(0)
            epoch_seen += bs
            # Track raw loss components averaged per-sample (unscaled by accum)
            epoch_loss += loss_raw.item() * bs
            epoch_ce   += ce.item() * bs
            epoch_cons += cons.item() * bs
            epoch_coral+= coral.item() * bs
            epoch_reg  += reg.item() * bs

            if (opt_step % cfg.log_every) == 0 and do_step:
                lr0 = optimizer.param_groups[0]["lr"]
                writer.add_scalar("train/loss", loss_raw.item(), opt_step)
                writer.add_scalar("train/ce", ce.item(), opt_step)
                writer.add_scalar("train/consistency", cons.item(), opt_step)
                writer.add_scalar("train/coral", coral.item(), opt_step)
                writer.add_scalar("train/reg", reg.item(), opt_step)
                writer.add_scalar("train/lr_head", lr0, opt_step)
                if len(optimizer.param_groups) > 1:
                    writer.add_scalar("train/lr_backbone", optimizer.param_groups[1]["lr"], opt_step)
                writer.flush()
                if (opt_step % 500) == 0:
                    print(f"[tb] wrote scalars at opt_step {opt_step} to {cfg.out_dir}")

            pbar.set_postfix({
                "loss": f"{epoch_loss/epoch_seen:.4f}",
                "ce":   f"{epoch_ce/epoch_seen:.4f}",
                "cons": f"{epoch_cons/epoch_seen:.4f}",
                "coral":f"{epoch_coral/epoch_seen:.4f}",
                "reg":  f"{epoch_reg/epoch_seen:.4f}",
                "Cs":   Cs_last
            })

        # End epoch: evaluate
        if ((epoch+1) % cfg.eval_every_epochs) == 0:
            r1_full, r1_val, tpr1e2, tpr1e3 = evaluate(val_loader, model_s, W_full, W_val, val_global_indices,
                                                       cfg, writer, opt_step, device)
            print(f"[eval] epoch {epoch+1} | R@1(full) {r1_full:.4f} | R@1(valGallery) {r1_val:.4f} | TPR@1e-2 {tpr1e2:.4f} | TPR@1e-3 {tpr1e3:.4f}")

            # Save best (by val-only gallery)
            if cfg.save_best and r1_val > best_recall_val:
                best_recall_val = r1_val
                ckpt = {
                    "cfg": vars(cfg),
                    "model": model_s.state_dict(),
                    "model_ema": model_ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch+1,
                    "global_step": opt_step,
                    "best_recall_val": best_recall_val,
                    "id_to_index": id_to_index,
                    "index_to_id": index_to_id,
                }
                torch.save(ckpt, os.path.join(cfg.out_dir, "best.pth"))
                print(f"[info] Saved best checkpoint (R@1 valGallery={best_recall_val:.4f})")

        # Save last each epoch
        ckpt = {
            "cfg": vars(cfg),
            "model": model_s.state_dict(),
            "model_ema": model_ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch+1,
            "global_step": opt_step,
            "best_recall_val": best_recall_val,
            "id_to_index": id_to_index,
            "index_to_id": index_to_id,
        }
        torch.save(ckpt, os.path.join(cfg.out_dir, "last.pth"))

        dt = time.time() - t0
        print(f"[info] epoch {epoch+1} done in {dt/60:.1f} min | avg loss {epoch_loss/epoch_seen:.4f}")

    writer.close()
    print("[done] Training completed.")

if __name__ == "__main__":
    set_seed(CFG.seed)
    main(CFG)
