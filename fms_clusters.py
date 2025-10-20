#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import torch
from collections import OrderedDict
from sklearn.cluster import KMeans
import random

from torchvision import transforms

IMAGES_DIR_DEFAULT = "/blue/azare/samgallic/Research/MorphExperiments/data/leaves/images"
OUT_DIR_DEFAULT    = "/blue/azare/samgallic/Research/MorphExperiments/data/leaves/10_clusters"
CKPT_DEFAULT       = "pytorchcifar/checkpoint/ckpt.pth"

# --- YOUR TRANSFORMS (kept as-is) ---
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.RandomRotation((0, 360))
])

def read_metadata_expand_ranges(csv_path, suffix=".jpg"):
    df = pd.read_csv(csv_path)
    filename_to_meta = {}
    for _, row in df.iterrows():
        start, end = map(int, str(row["filename"]).split("-"))
        for i in range(start, end + 1):
            fname = f"{i}{suffix}"
            filename_to_meta[fname] = {
                "label": int(row["label"]),
                "scientific_name": row.get("Scientific Name", ""),
                "common_name": row.get("Common Name(s)", ""),
                "url": row.get("URL", ""),
            }
    all_filenames = sorted(filename_to_meta.keys())
    return all_filenames, filename_to_meta

def build_resnet50_pytorchcifar(ckpt_path, device):
    from pytorchcifar.models import resnet
    model = resnet.ResNet50()
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("net", ckpt)
        if any(k.startswith("module.") for k in state.keys()):
            state = OrderedDict((k.replace("module.", ""), v) for k, v in state.items())
        model.load_state_dict(state, strict=True)
        print(f"✓ loaded checkpoint: {ckpt_path}")
    model.to(device).eval()
    return model

def load_tensor_with_transform(path):
    img = Image.open(path).convert("RGB")
    t = transform(img)  # [3,128,128] after your pipeline
    return t

def main():
    ap = argparse.ArgumentParser(description="KMeans(10) on POS=10 using your transforms; save 10 layer4 feature maps.")
    ap.add_argument("--labels_csv", required=True, help="Path to labels.csv (ranged 'filename').")
    ap.add_argument("--images_dir", default=IMAGES_DIR_DEFAULT)
    ap.add_argument("--out_dir", default=OUT_DIR_DEFAULT)
    ap.add_argument("--ckpt", default=CKPT_DEFAULT)
    ap.add_argument("--pos_label", type=int, default=10)
    ap.add_argument("--suffix", default=".jpg")
    args = ap.parse_args()

    # reproducibility (RandomRotation uses torch RNG)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs(args.out_dir, exist_ok=True)
    feat_dir = os.path.join(args.out_dir, "features")
    os.makedirs(feat_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Filter POS=10 images present on disk
    all_fns, meta = read_metadata_expand_ranges(args.labels_csv, suffix=args.suffix)
    pos_fns = [f for f in all_fns if meta[f]["label"] == args.pos_label]
    pos_fns = [f for f in pos_fns if os.path.exists(os.path.join(args.images_dir, f))]
    if not pos_fns:
        raise RuntimeError(f"No images with label=={args.pos_label} found in {args.images_dir}")

    print(f"Found {len(pos_fns)} images with label == {args.pos_label}")

    # 2) Build matrix for KMeans from YOUR transformed tensors
    X = []
    tensors = []  # keep per-image tensor to avoid recomputing
    for f in pos_fns:
        t = load_tensor_with_transform(os.path.join(args.images_dir, f))  # [3,128,128]
        tensors.append(t)
        X.append(t.view(-1).numpy())
    X = np.stack(X, axis=0)

    # 3) KMeans -> 10 clusters
    kmeans = KMeans(n_clusters=10, n_init="auto", random_state=42)
    labels = kmeans.fit_predict(X)  # [N]
    print("✓ KMeans finished")

    # 4) Average transformed tensor per cluster (still in your normalized space)
    cluster_avgs = []
    for cid in range(10):
        idx = np.where(labels == cid)[0]
        if idx.size == 0:
            print(f"Cluster {cid:02d} empty; skipping.")
            cluster_avgs.append(None)
            continue
        stack = torch.stack([tensors[i] for i in idx], dim=0)  # [n_c,3,128,128]
        avg_t = stack.mean(dim=0)                              # [3,128,128]
        cluster_avgs.append(avg_t)

    # 5) Hook layer4 and save each average image’s feature map
    model = build_resnet50_pytorchcifar(args.ckpt, device)
    children = dict(model.named_children())
    feature_module = children.get("layer4", None)
    if feature_module is None:
        feature_module = list(model.children())[-2]
        print("⚠️ 'layer4' not found by name; using penultimate child for features.")

    feature_blob = {}
    def _hook(m, i, o):
        feature_blob["out"] = o.detach().cpu().squeeze(0)  # [C,H',W']
    handle = feature_module.register_forward_hook(_hook)

    saved = 0
    with torch.no_grad():
        for cid, avg_t in enumerate(cluster_avgs):
            if avg_t is None:
                continue
            x = avg_t.unsqueeze(0).to(device)  # [1,3,128,128] already normalized/rotated/resized per your transform
            _ = model(x)                        # trigger hook
            feat = feature_blob.get("out", None)
            if feat is None:
                print(f"skip cluster {cid:02d}: no features captured")
                continue
            out_path = os.path.join(feat_dir, f"cluster_{cid:02d}.pt")
            torch.save(feat, out_path)
            saved += 1
            print(f"✓ cluster {cid:02d} → feature {tuple(feat.shape)} saved: {out_path}")

    handle.remove()
    print(f"\nDone. Saved {saved} feature maps to: {feat_dir}")

if __name__ == "__main__":
    main()
