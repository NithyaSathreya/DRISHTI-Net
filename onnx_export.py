"""
onnx_export.py — Export generator to ONNX (attention_unet or gan_unet)
"""

import argparse
import os
import torch
from config import load_config
from models.model_factory import build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  type=str, default="configs/baseline.yaml")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--output",  type=str, default="msarnet.onnx")
    parser.add_argument("--opset",   type=int, default=17)
    return parser.parse_args()


def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    device = torch.device("cpu")

    model      = build_model({"model": vars(cfg.model)})
    model_type = getattr(cfg.model, "type", "attention_unet")

    state_dict = torch.load(args.weights, map_location=device)
    if model_type == "gan_unet":
        model.generator.load_state_dict(state_dict)
        export_model = model.generator
    else:
        model.load_state_dict(state_dict)
        export_model = model

    export_model.eval()

    ir_size   = getattr(cfg.dataset, "ir_size",  256)
    dummy     = torch.randn(1, cfg.model.in_channels, ir_size, ir_size)

    print(f"Exporting {model_type} generator -> {args.output}")
    torch.onnx.export(
        export_model, dummy, args.output,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["ir"],
        output_names=["rgb"],
        dynamic_axes={"ir": {0: "batch"}, "rgb": {0: "batch"}}
    )
    print(f"Saved: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()