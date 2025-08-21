import os
import sys
import glob
from typing import List

import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn as nn
import torchvision.transforms as T

import onnx
from onnx2torch import convert

import insightface
from insightface.utils import face_align as fa

import open_clip
from tqdm import tqdm

import matplotlib.pyplot as plt


# ---------------------------
# CONFIG (can edit)
# ---------------------------
SKETCH_PATH = r"F:\FaceSketch\full\sketch\4bcae0de-48b1-40c2-8b9f-1625a9419c7f.png"  # "./sketch.jpg"
FFHQ_DIR = r"F:\ffhq"
MAX_GALLERY = 100
CKPT_DIR = os.path.join("runs", "sketch2arcfacev2")
PREFER_BEST = True


# ---------------------------
# UTILITIES / MODELS (mirrors train.py)
# ---------------------------
def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(p=2, dim=1, keepdim=True).clamp_min(eps)


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


def build_sketch_eval_transform(image_size: int = 224):
    # CLIP mean/std
    mean = (0.48145466, 0.4578275, 0.40821073)
    std  = (0.26862954, 0.26130258, 0.27577711)
    eval_tx = T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return eval_tx


# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def find_checkpoint(ckpt_dir: str, prefer_best: bool = True) -> str:
    best_path = os.path.join(ckpt_dir, "best.pth")
    last_path = os.path.join(ckpt_dir, "last.pth")
    if prefer_best and os.path.isfile(best_path):
        return best_path
    if os.path.isfile(last_path):
        return last_path
    if os.path.isfile(best_path):
        return best_path
    raise FileNotFoundError(f"No checkpoint found in {ckpt_dir} (expected best.pth or last.pth)")


def load_sketch_encoder_from_ckpt(ckpt_path: str, device: torch.device) -> (SketchEncoder, dict):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("cfg", {})
    clip_model = cfg.get("clip_model", "ViT-B-16")
    clip_pretrained = cfg.get("clip_pretrained", "openai")
    proj_hidden = cfg.get("proj_hidden", 1024)
    embed_dim = cfg.get("embed_dim", 512)
    model = SketchEncoder(clip_model, clip_pretrained, proj_hidden, embed_dim, freeze_backbone=True).to(device)
    model.load_state_dict(ckpt["model"])  # finetuned part
    model.eval()
    return model, cfg


def list_first_n_images(folder: str, n: int, exts=("*.png", "*.jpg", "*.jpeg", "*.webp")) -> List[str]:
    files: List[str] = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(folder, ext)))
    files = sorted(files)
    return files[:n]


def main():
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load finetuned SketchEncoder
    ckpt_path = find_checkpoint(CKPT_DIR, PREFER_BEST)
    print(f"[info] Using checkpoint: {ckpt_path}")
    model_s, cfg = load_sketch_encoder_from_ckpt(ckpt_path, device)

    # 2) Build eval transform for sketch encoding (must match train.py eval)
    eval_tx = build_sketch_eval_transform(image_size=224)

    # 3) Load and embed the sketch
    sketch_img = Image.open(SKETCH_PATH).convert("RGB")
    x_sketch = eval_tx(sketch_img).unsqueeze(0).to(device)
    with torch.no_grad():
        z_sketch = model_s(x_sketch).squeeze(0)  # (512,) already L2-normalized

    # Save sketch embedding (similar to ArcFaceEmbeddingSave.py)
    sketch_embedding_path = "sketch_embedding.npy"
    z_sketch_cpu = z_sketch.detach().cpu().numpy()
    np.save(sketch_embedding_path, z_sketch_cpu)
    print(f"[info] Saved sketch embedding to: {sketch_embedding_path}")
    print(f"[info] Sketch embedding shape: {z_sketch_cpu.shape}, L2 norm: {np.linalg.norm(z_sketch_cpu):.6f}")

    # 4) Load ArcFace teacher to embed real photos
    arcface_onnx = cfg.get("arcface_onnx", r"D:\\hf\\hub\\w600k_r50.onnx")
    if not os.path.isfile(arcface_onnx):
        raise FileNotFoundError(f"ArcFace ONNX not found at: {arcface_onnx}")
    arc = ArcFaceW600kR50(arcface_onnx, device=str(device))

    # 5) Collect first MAX_GALLERY real images
    gallery_paths = list_first_n_images(FFHQ_DIR, MAX_GALLERY)
    if len(gallery_paths) == 0:
        raise RuntimeError(f"No images found in {FFHQ_DIR}")
    print(f"[info] Found {len(gallery_paths)} gallery images (showing first {min(MAX_GALLERY, len(gallery_paths))}).")

    # 6) Align, preprocess and embed gallery with ArcFace
    embeds = []
    for p in tqdm(gallery_paths, desc="Embed gallery", ncols=100):
        img = Image.open(p).convert("RGB")
        img_aligned = align_photo(img)
        x = preprocess_for_arcface(img_aligned).to(arc.device)
        with torch.no_grad():
            e = arc(x).squeeze(0)  # (512,)
        embeds.append(e)
    E = torch.stack(embeds, dim=0)  # (N,512), L2-normalized

    # 7) Compare (cosine similarity) sketch vs each real embedding
    z = z_sketch.to(E.device)
    sims = (E @ z)  # (N,)
    sims_cpu = sims.detach().cpu().numpy()
    order = np.argsort(-sims_cpu)  # descending

    # 8) Report top-5 and plot results
    top_k = min(5, len(gallery_paths))
    print("\nTop matches (cosine similarity):")
    for rank in range(top_k):
        idx = int(order[rank])
        print(f"{rank+1:2d}. {gallery_paths[idx]}\t sim={sims_cpu[idx]:.4f}")

    # 9) Plot sketch and top-5 matches
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle("Sketch Recognition Results", fontsize=14, fontweight='bold')

    # Plot sketch as reference (top-left)
    axes[0, 0].imshow(sketch_img)
    axes[0, 0].set_title("Query Sketch", fontsize=12)
    axes[0, 0].axis('off')

    # Hide unused subplots in first row
    for col in range(1, 5):
        axes[0, col].set_visible(False)

    # Plot top-5 matches in second row
    for rank in range(top_k):
        idx = int(order[rank])
        match_path = gallery_paths[idx]
        similarity = sims_cpu[idx]

        # Load and display the matched image
        match_img = Image.open(match_path).convert("RGB")
        axes[1, rank].imshow(match_img)

        # Extract filename from path for cleaner display
        filename = os.path.basename(match_path)
        axes[1, rank].set_title(f"#{rank+1}: {filename}\nSimilarity: {similarity:.4f}", fontsize=10)
        axes[1, rank].axis('off')

    # Hide any unused subplots in second row
    for rank in range(top_k, 5):
        axes[1, rank].set_visible(False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
