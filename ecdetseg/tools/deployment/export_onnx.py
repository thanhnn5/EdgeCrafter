"""
EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
---------------------------------------------------------------------------------
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))

import torch
import torch.nn as nn

from engine.core import YAMLConfig


def main(args, ):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)
    
    task = cfg.yaml_cfg['task']

    if args.resume:
        # Backbone weights will be overwritten by the resume checkpoint;
        # skip the default backbone-load to avoid a redundant download.
        if 'ViTAdapter' in cfg.yaml_cfg:
            cfg.yaml_cfg['ViTAdapter']['skip_load_backbone'] = True
        if 'ConvNeXtAdapter' in cfg.yaml_cfg:
            cfg.yaml_cfg['ConvNeXtAdapter']['pretrained'] = False
        checkpoint = torch.load(args.resume, map_location='cpu')

    # Export-time override: swap exact GELU (ONNX:Erf) for tanh-approx GELU
    # (ONNX:Tanh) when targeting backends without Erf support (e.g. MNN Metal).
    if args.gelu_tanh and 'ConvNeXtAdapter' in cfg.yaml_cfg:
        cfg.yaml_cfg['ConvNeXtAdapter']['gelu_approximate'] = 'tanh'
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']

        # NOTE load train mode state -> convert to deploy mode
        cfg.model.load_state_dict(state)

    else:
        # raise AttributeError('Only support resume to load model.state_dict by now.')
        print('not load model.state_dict, use default init state dict...')

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()

        def forward(self, images):
            outputs = self.model(images)
            return outputs['pred_logits'], outputs['pred_boxes']

    model = Model()
    model = model.cpu()
    model.eval()

    img_size = cfg.yaml_cfg["eval_spatial_size"]
    data = torch.rand(1, 3, *img_size)
    size = torch.tensor([img_size])
    _ = model(data)

    dynamic_axes = None

    output_file = args.resume.replace('.pth', '.onnx') if args.resume else 'model.onnx'
    
    torch.onnx.export(
        model,
        (data,),
        output_file,
        input_names=['images'],
        output_names=['pred_logits', 'pred_boxes'],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        verbose=False,
        do_constant_folding=True,
    )

    if args.check:
        import onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)
        print('Check export onnx model done...')

    if args.simplify:
        import onnx
        import onnxsim
        input_shapes = {'images': data.shape}
        onnx_model_simplify, check = onnxsim.simplify(output_file, test_input_shapes=input_shapes)
        onnx.save(onnx_model_simplify, output_file)
        print(f'Simplify onnx model {check}...')


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='configs/dfine/dfine_hgnetv2_l_coco.yml', type=str, )
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--opset', type=int, default=18,)
    parser.add_argument('--check',  action='store_true')
    parser.add_argument('--simplify',  action='store_true')
    parser.add_argument('--gelu-tanh', action='store_true',
                        help='Swap exact GELU for tanh-approximate GELU before export. '
                             'Avoids the ONNX Erf op for backends that lack it '
                             '(e.g. MNN Metal). Only affects ConvNeXtAdapter configs.')
    args = parser.parse_args()
    main(args)
