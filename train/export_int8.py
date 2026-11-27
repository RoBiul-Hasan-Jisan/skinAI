

import argparse
from pathlib import Path

import torch

from model.bd_skinnet import BDSkinNet, IMAGE_SIZE
from model.disease_labels import NUM_CLASSES


def export_int8(checkpoint_path: str, output_path: str = None):
    if output_path is None:
        output_path = str(Path(checkpoint_path).parent / "bd_skinnet_int8.pth")

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    class_to_idx = ckpt.get("class_to_idx")
    num_classes = int(ckpt.get("num_classes", len(class_to_idx) if class_to_idx else NUM_CLASSES))

    model = BDSkinNet(num_classes=num_classes)
    model.load_state_dict(ckpt.get("model_state", ckpt))
    model.eval()

    print(f"FP32 parameters: {sum(p.numel() for p in model.parameters()):,}")

    quantized = torch.quantization.quantize_dynamic(
        model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8
    )

    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        out = quantized(dummy)
    assert out.shape == (1, num_classes), f"Shape mismatch: {out.shape}"
    print(f"INT8 output shape: {out.shape} - OK")

    torch.save({
        "model_state":  quantized.state_dict(),
        "num_classes":  num_classes,
        "class_to_idx": class_to_idx,
        "quantized":    True,
        "val_f1":       ckpt.get("val_f1"),
        "epoch":        ckpt.get("epoch"),
    }, output_path)

    in_mb = Path(checkpoint_path).stat().st_size / 1e6
    out_mb = Path(output_path).stat().st_size / 1e6
    print(f"Saved INT8 model: {output_path}")
    print(f"Size: {in_mb:.1f} MB -> {out_mb:.1f} MB ({100 * (1 - out_mb / in_mb):.0f}% reduction)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    export_int8(args.checkpoint, args.output)
