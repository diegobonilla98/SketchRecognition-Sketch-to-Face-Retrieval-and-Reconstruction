import os
import cv2
import onnx
import torch
import numpy as np
from PIL import Image
from onnx2torch import convert

from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from transformers import AutoFeatureExtractor
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from huggingface_hub import hf_hub_download

from ip_adapter.ip_adapter_faceid import IPAdapterFaceID, IPAdapterFaceIDPlus


faceid_embed_npy = "sketch_embedding.npy"  # "00064_embed.npy"  # alternatively, provide path to a precomputed ArcFace embedding (.npy). Use this OR images, not both.
plus_face_image_path = r"F:\FaceSketch\full\sketch\4bcae0de-48b1-40c2-8b9f-1625a9419c7f.png"  # None  # optional: if using faceid_embed_npy and you want FaceID Plus structure guidance, provide a face image path here
person_description = "Adult Asian male, 25–35, medium build, oval face with defined cheekbones, straight brows, narrow eyes, medium nose, thin lips, medium-length dark wavy hair, clean-shaven, ears slightly protruding."
prompt = f"A photo of {person_description.lower()}. Looking at the camera, neutral expression, white background, ID photo."
negative_prompt = "low quality, extra fingers, extra limbs, deformed, b&w, drawing, sketch, cartoon"
preserve_face_structure = True          # True -> use IP-Adapter FaceID Plus
face_strength = 1.6                     # only used when preserve_face_structure=True
likeness_strength = 1.1                 # controls how strongly the face embedding guides
nfaa_negative_prompt = ""

# Paths/models
base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

# IP-Adapter FaceID weights (SD1.5 variants)
ip_ckpt = hf_hub_download(repo_id="h94/IP-Adapter-FaceID", filename="ip-adapter-faceid_sd15.bin", repo_type="model")
ip_plus_ckpt = hf_hub_download(repo_id="h94/IP-Adapter-FaceID", filename="ip-adapter-faceid-plusv2_sd15.bin", repo_type="model")

# Path to the exact ArcFace ONNX from buffalo_l: w600k_r50.onnx
# (Put the file in your working dir or point to the correct path)
arcface_onnx_path = r"D:\hf\hub\w600k_r50.onnx"

# Output
out_path = "ip_adapter_output.png"

# -----------------------------
# Utilities: ArcFace (PyTorch)
# -----------------------------

def preprocess_for_arcface(img_rgb: np.ndarray) -> torch.Tensor:
    """
    Expect RGB uint8 image. Resize to 112x112 and map to [-1, 1].
    Returns FloatTensor (1,3,112,112).
    """
    chip = cv2.resize(img_rgb, (112, 112), interpolation=cv2.INTER_LINEAR)
    chip = chip.astype(np.float32)
    chip = (chip - 127.5) / 127.5
    chip = np.transpose(chip, (2, 0, 1))  # CHW
    return torch.from_numpy(chip).unsqueeze(0)

def l2n(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(p=2, dim=1, keepdim=True).clamp_min(eps)

class ArcFaceW600kR50Torch(torch.nn.Module):
    """
    Exact buffalo_l recognition backbone (w600k_r50.onnx) converted to PyTorch.
    Forward takes either a preprocessed tensor (N,3,112,112) or raw RGB np arrays/PIL/paths.
    Returns 512-D L2-normalized embeddings (N,512).
    """
    def __init__(self, onnx_path: str, device: str = "cuda"):
        super().__init__()
        model_onnx = onnx.load(onnx_path)
        self.core = convert(model_onnx)
        self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        self.to(self.device)
        self.eval()

    @torch.no_grad()
    def forward(self, inputs):
        # Accept: path / PIL.Image / np.ndarray (RGB) / list thereof / preprocessed tensor
        if torch.is_tensor(inputs):
            x = inputs  # assume preprocessed (N,3,112,112) in [-1,1]
        else:
            imgs = inputs if isinstance(inputs, (list, tuple)) else [inputs]
            batch = []
            for im in imgs:
                if isinstance(im, str):
                    im = Image.open(im).convert("RGB")
                if isinstance(im, Image.Image):
                    im = np.array(im)  # RGB
                elif isinstance(im, np.ndarray):
                    # assume already RGB
                    if im.ndim != 3 or im.shape[2] != 3:
                        raise ValueError("Expected RGB HxWx3 array.")
                else:
                    raise TypeError(f"Unsupported input type: {type(im)}")
                batch.append(preprocess_for_arcface(im))
            x = torch.cat(batch, dim=0)
        out = self.core(x.to(self.device))
        if isinstance(out, (list, tuple)):
            out = out[0]
        return l2n(out).float().cpu()

# -----------------------------
# Simple face crops for Plus
# -----------------------------

def naive_square_center_crop(image_rgb: np.ndarray, out_size: int) -> Image.Image:
    """
    Minimal cropping used for IP-Adapter Plus face_image when you don't have landmarks.
    If your inputs are already portrait-style with centered faces, this works decently.
    """
    h, w, _ = image_rgb.shape
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    crop = image_rgb[y0:y0+s, x0:x0+s]
    crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    return Image.fromarray(crop)

# -----------------------------
# Build diffusion pipeline
# -----------------------------

def build_sd15_pipe(device: str = "cuda"):
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
    safety_model_id = "CompVis/stable-diffusion-safety-checker"
    safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
    safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        vae=vae,
        feature_extractor=safety_feature_extractor,
        safety_checker=safety_checker
    ).to(device)
    return pipe

# -----------------------------
# Main generation
# -----------------------------

def generate_with_ip_adapter(
    image_paths,
    prompt,
    negative_prompt,
    preserve_face_structure,
    face_strength,
    likeness_strength,
    nfaa_negative_prompt,
    arcface_onnx_path,
    device="cuda",
    out_path="ip_adapter_output.png",
    faceid_embed_npy=None,
    plus_face_image_path=None
):
    # Reproducibility tweaks
    torch.set_grad_enabled(False)
    cv2.setNumThreads(1)

    # Validate inputs (mutually exclusive)
    if faceid_embed_npy is not None and image_paths:
        raise ValueError("Provide either image_paths or faceid_embed_npy, not both.")
    if (faceid_embed_npy is None) and (not image_paths or len(image_paths) == 0):
        raise ValueError("You must provide image_paths (one or more) or a faceid_embed_npy path.")

    using_npy_embed = faceid_embed_npy is not None

    # 1) Build SD pipeline
    pipe = build_sd15_pipe(device=device)

    # 2) Load IP-Adapter
    ip_model = IPAdapterFaceID(pipe, ip_ckpt, device)
    ip_model_plus = IPAdapterFaceIDPlus(pipe, image_encoder_path, ip_plus_ckpt, device)

    face_image_for_plus = None
    if not using_npy_embed:
        # 3) Load ArcFace (PyTorch) and compute embeddings for each image
        arcface = ArcFaceW600kR50Torch(arcface_onnx_path, device=device)

        faceid_embeds = []
        first = True
        for p in image_paths:
            # Read as RGB for both embedding and optional Plus face image
            img_rgb = np.array(Image.open(p).convert("RGB"))
            emb = arcface(img_rgb)              # (1,512) CPU float32, L2-normalized
            faceid_embeds.append(emb)

            if first and preserve_face_structure:
                # Build a naive 224x224 centered portrait crop for the "Plus" model
                face_image_for_plus = naive_square_center_crop(img_rgb, out_size=224)
                first = False

        # 4) Average embeddings (then re-normalize)
        avg_embed = torch.mean(torch.cat(faceid_embeds, dim=0), dim=0, keepdim=True)
        avg_embed = l2n(avg_embed).float()      # (1,512) CPU float32
    else:
        # 3) Load from .npy and prepare embedding
        arr = np.load(faceid_embed_npy, allow_pickle=False)
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr[None, :]
        elif arr.ndim != 2:
            raise ValueError("faceid_embed_npy must be shape (512,) or (N,512)")
        if arr.shape[1] != 512:
            raise ValueError(f"Expected embedding dimension 512, got {arr.shape[1]}")
        if arr.shape[0] > 1:
            arr = arr.mean(axis=0, keepdims=True)
        avg_embed = l2n(torch.from_numpy(arr.astype(np.float32)))  # (1,512) CPU float32
        # Optionally enable Plus with an explicit face image path (structure guidance)
        if preserve_face_structure and plus_face_image_path is not None:
            plus_img_rgb = np.array(Image.open(plus_face_image_path).convert("RGB"))
            face_image_for_plus = naive_square_center_crop(plus_img_rgb, out_size=224)
            # Optional: cosine similarity between provided .npy and this image's embedding for debugging
            try:
                arcface = ArcFaceW600kR50Torch(arcface_onnx_path, device=device)
                plus_emb = arcface(plus_img_rgb).numpy()  # (1,512), already L2-normalized
                cos_sim = float((plus_emb @ avg_embed.numpy().T)[0, 0])
                print(f"[Debug] Cosine similarity between .npy and plus image embedding: {cos_sim:.4f}")
            except Exception as e:
                print(f"[Warn] Could not compute cosine similarity for debug: {e}")
        elif preserve_face_structure and plus_face_image_path is None:
            print("Embedding-only input provided without plus_face_image_path; disabling FaceID Plus (needs a face image for structure).")
            preserve_face_structure = False

    # 5) Compose negative prompt
    total_negative_prompt = f"{negative_prompt} {nfaa_negative_prompt}".strip()

    # 6) Generate
    if not preserve_face_structure:
        print("Generating with FaceID (v1)")
        image = ip_model.generate(
            prompt=prompt,
            negative_prompt=total_negative_prompt,
            faceid_embeds=avg_embed,        # torch.FloatTensor (1,512)
            scale=likeness_strength,
            width=512,
            height=512,
            num_inference_steps=30
        )
    else:
        print("Generating with FaceID Plus (v2)")
        if face_image_for_plus is None:
            # Fallback: use first image, resized to 224
            face_image_for_plus = naive_square_center_crop(np.array(Image.open(image_paths[0]).convert("RGB")), 224)

        image = ip_model_plus.generate(
            prompt=prompt,
            negative_prompt=total_negative_prompt,
            faceid_embeds=avg_embed,        # torch.FloatTensor (1,512)
            scale=likeness_strength,
            face_image=face_image_for_plus, # PIL.Image (224x224)
            shortcut=True,
            s_scale=face_strength,
            width=512,
            height=512,
            num_inference_steps=30
        )

    # 7) Save/return
    if isinstance(image, Image.Image):
        image.save(out_path)
    else:
        # Many IP-Adapter demos return PIL directly; handle list/np just in case
        if isinstance(image, list) and len(image) > 0 and isinstance(image[0], Image.Image):
            image[0].save(out_path)
        else:
            Image.fromarray(np.asarray(image)).save(out_path)

    print(f"Saved: {out_path}")
    return out_path

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    generate_with_ip_adapter(
        image_paths=[],
        prompt=prompt,
        negative_prompt=negative_prompt,
        preserve_face_structure=preserve_face_structure,
        face_strength=face_strength,
        likeness_strength=likeness_strength,
        nfaa_negative_prompt=nfaa_negative_prompt,
        arcface_onnx_path=arcface_onnx_path,
        device="cuda",
        out_path=out_path,
        faceid_embed_npy=faceid_embed_npy,
        plus_face_image_path=plus_face_image_path
    )
