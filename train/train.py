

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from model.bd_skinnet import IMAGE_SIZE, MEAN, STD, BDSkinNet, inference_transform

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# Data
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def scan_folder(root: Path, class_to_idx: dict | None = None):
    """Return (samples, class_to_idx). samples = [(path, label_idx), ...]."""
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    if class_to_idx is None:
        class_to_idx = {c: i for i, c in enumerate(classes)}
    samples = []
    for c in classes:
        if c not in class_to_idx:
            print(f"[warn] {root}/{c} not in main dataset classes - skipped")
            continue
        for p in sorted((root / c).rglob("*")):
            if p.suffix.lower() in IMG_EXTS:
                samples.append((str(p), class_to_idx[c]))
    return samples, class_to_idx


def stratified_split(samples, train=0.70, val=0.15, seed=42):
    """Image-level stratified split (the source datasets have no patient IDs)."""
    rng = random.Random(seed)
    by_class: dict[int, list] = {}
    for s in samples:
        by_class.setdefault(s[1], []).append(s)
    tr, va, te = [], [], []
    for items in by_class.values():
        rng.shuffle(items)
        n = len(items)
        n_tr = int(round(n * train))
        n_va = int(round(n * val))
        tr += items[:n_tr]
        va += items[n_tr:n_tr + n_va]
        te += items[n_tr + n_va:]
    return tr, va, te


def train_transform():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.5),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


class SkinDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"cannot read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(image=img)["image"], label


# Metrics (no sklearn dependency)
def confusion_matrix(y_true, y_pred, n):
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def metrics_from_cm(cm):
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return {
        "accuracy": float(tp.sum() / max(cm.sum(), 1)),
        "macro_f1": float(f1.mean()),
        "per_class": {"precision": prec.tolist(), "recall": rec.tolist(), "f1": f1.tolist()},
    }


# Model / optimisation
def build_model(num_classes: int, pretrained: bool, device: str) -> BDSkinNet:
    model = BDSkinNet(num_classes=num_classes)
    if pretrained:
        # BDSkinNet builds the backbone with pretrained=False, so load ImageNet
        # weights into it here (same features_only structure -> same keys).
        ref = timm.create_model(
            "swin_base_patch4_window7_224", pretrained=True,
            features_only=True, out_indices=(0, 1, 2, 3),
        )
        missing, unexpected = model.swin.load_state_dict(ref.state_dict(), strict=False)
        print(f"[pretrained] loaded ImageNet weights "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
        del ref
    return model.to(device)


def make_optimizer(model, lr_backbone, lr_head, weight_decay):
    backbone = list(model.swin.parameters())
    head = list(model.cbam_modules.parameters()) + list(model.head.parameters())
    return torch.optim.AdamW(
        [{"params": backbone, "lr": lr_backbone, "base_lr": lr_backbone},
         {"params": head, "lr": lr_head, "base_lr": lr_head}],
        weight_decay=weight_decay,
    )


def set_lr(optimizer, step, total_steps, warmup_steps):
    for g in optimizer.param_groups:
        if step < warmup_steps:
            scale = (step + 1) / warmup_steps
        else:
            prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            scale = 0.5 * (1 + math.cos(math.pi * prog))
        g["lr"] = g["base_lr"] * max(scale, 0.01)


@torch.no_grad()
def evaluate(model, loader, device, num_classes, criterion=None):
    model.eval()
    ys, ps, loss_sum, n = [], [], 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if criterion is not None:
            loss_sum += criterion(logits, y).item() * len(y)
        n += len(y)
        ys += y.cpu().tolist()
        ps += logits.argmax(1).cpu().tolist()
    cm = confusion_matrix(ys, ps, num_classes)
    m = metrics_from_cm(cm)
    m["loss"] = loss_sum / max(n, 1)
    m["confusion_matrix"] = cm.tolist()
    return m


# Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--synthetic_dir", default=None, help="extra images, TRAIN split only")
    ap.add_argument("--out", default="model/checkpoints/bd_skinnet_best.pth")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr_backbone", type=float, default=2e-5)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--warmup_epochs", type=int, default=2)
    ap.add_argument("--patience", type=int, default=10, help="early-stop on val macro-F1")
    ap.add_argument("--balanced_sampler", action="store_true",
                    help="oversample minority classes instead of loss weighting")
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # ---- data
    samples, class_to_idx = scan_folder(Path(args.data_dir))
    if not samples:
        raise SystemExit(f"no images found under {args.data_dir}")
    num_classes = len(class_to_idx)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    print(f"classes ({num_classes}): {class_to_idx}")

    tr, va, te = stratified_split(samples, seed=args.seed)
    if args.synthetic_dir:
        synth, _ = scan_folder(Path(args.synthetic_dir), class_to_idx)
        tr += synth
        print(f"added {len(synth)} synthetic images to the train split only")
    print(f"split sizes  train={len(tr)}  val={len(va)}  test={len(te)}")

    counts = np.bincount([l for _, l in tr], minlength=num_classes).astype(float)
    print("train class counts:", {idx_to_class[i]: int(c) for i, c in enumerate(counts)})

    train_ds = SkinDataset(tr, train_transform())
    val_ds = SkinDataset(va, inference_transform)
    test_ds = SkinDataset(te, inference_transform)

    pin = device == "cuda"
    if args.balanced_sampler:
        w = (1.0 / np.maximum(counts, 1))[[l for _, l in tr]]
        sampler = WeightedRandomSampler(w, num_samples=len(tr), replacement=True)
        train_loader = DataLoader(train_ds, args.batch_size, sampler=sampler,
                                  num_workers=args.workers, pin_memory=pin, drop_last=True)
        class_w = None
    else:
        train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                                  num_workers=args.workers, pin_memory=pin, drop_last=True)
        class_w = torch.tensor(counts.sum() / (num_classes * np.maximum(counts, 1)),
                               dtype=torch.float32, device=device)
    val_loader = DataLoader(val_ds, args.batch_size * 2, num_workers=args.workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, args.batch_size * 2, num_workers=args.workers, pin_memory=pin)

    # ---- model
    model = build_model(num_classes, pretrained=not args.no_pretrained, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_w, label_smoothing=args.label_smoothing)
    eval_criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, args.lr_backbone, args.lr_head, args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    best_f1, best_epoch, bad, step, history = -1.0, 0, 0, 0, []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, run_loss, seen = time.time(), 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            set_lr(optimizer, step, total_steps, warmup_steps)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            run_loss += loss.item() * len(y)
            seen += len(y)
            step += 1

        v = evaluate(model, val_loader, device, num_classes, eval_criterion)
        row = {"epoch": epoch, "train_loss": run_loss / max(seen, 1),
               "val_loss": v["loss"], "val_acc": v["accuracy"], "val_macro_f1": v["macro_f1"],
               "seconds": round(time.time() - t0, 1)}
        history.append(row)
        print(f"epoch {epoch:3d}/{args.epochs}  train_loss {row['train_loss']:.4f}  "
              f"val_loss {row['val_loss']:.4f}  val_acc {row['val_acc']:.4f}  "
              f"val_F1 {row['val_macro_f1']:.4f}  ({row['seconds']}s)")

        if v["macro_f1"] > best_f1:
            best_f1, best_epoch, bad = v["macro_f1"], epoch, 0
            torch.save({
                "model_state": model.state_dict(),
                "class_to_idx": class_to_idx,       # read by model.bd_skinnet.load_model
                "num_classes": num_classes,
                "val_f1": best_f1,
                "epoch": epoch,
                "quantized": False,
            }, out_path)
            print(f"  saved best -> {out_path}")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stopping (no val F1 gain for {args.patience} epochs)")
                break

        (Path(args.results_dir) / "train_log.json").write_text(json.dumps(history, indent=2))

    # ---- final test evaluation with the best checkpoint
    ckpt = torch.load(out_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    t = evaluate(model, test_loader, device, num_classes, eval_criterion)
    t["classes"] = [idx_to_class[i] for i in range(num_classes)]
    t["best_epoch"] = best_epoch
    t["best_val_macro_f1"] = best_f1
    t["n_test"] = len(te)
    (Path(args.results_dir) / "test_metrics.json").write_text(json.dumps(t, indent=2))

    print(f"\nTEST  accuracy {t['accuracy']:.4f}   macro-F1 {t['macro_f1']:.4f}   (n={len(te)})")
    for i, name in enumerate(t["classes"]):
        print(f"  {name:24s} P {t['per_class']['precision'][i]:.3f}  "
              f"R {t['per_class']['recall'][i]:.3f}  F1 {t['per_class']['f1'][i]:.3f}")
    print("note: split is by image, not by patient - test score may be optimistic.")


if __name__ == "__main__":
    main()
