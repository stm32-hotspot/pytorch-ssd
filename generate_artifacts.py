# Copyright (c) 2025 STMicroelectronics. All rights reserved.
#

import argparse
from onnxruntime.training import artifacts
from config import optimized_grad
import onnxruntime.training.onnxblock as onnxblock
import onnx
import numpy as np

def main():
    parser = argparse.ArgumentParser(
        description='Single Shot MultiBox Detector Artifacts Generation With ORT')
    parser.add_argument('onnx_model_path',
                        help='Path of the SSD onnx model. The loss of the model should be included into the graph.')
    parser.add_argument('artifacts_dir_path',
                        help='Path of the directory in which to store the artifacts.')
    parser.add_argument('--freeze_net',
                        help='Freeze all layers but headers.', action="store_true")
    parser.add_argument('--freeze_basenet',
                        help='Freeze only basenet.', action="store_true")
    parser.add_argument('--freeze_optimized',
                        help='Freeze layers according to a specific mask. The mask consists in a list of layers indices to be updated.', action="store_true")
    args = parser.parse_args()

    # Load the onnx model.
    model_path = args.onnx_model_path
    onnx_model = onnx.load(model_path)

    # Remove running_var and running_mean outputs from bn nodes in order to avoid shape inference error
    # cf https://stackoverflow.com/questions/77486728/batchnorms-force-set-to-training-mode-on-torch-onnx-export-when-running-stats-ar
    for node in onnx_model.graph.node:
        if node.op_type == "BatchNormalization":
            for attribute in node.attribute:
                if attribute.name == 'training_mode':
                    if attribute.i == 1:
                        node.output.remove(node.output[1])
                        node.output.remove(node.output[1])
                    attribute.i = 0

    if args.freeze_optimized:
        requires_grad = [param.name for param in onnx_model.graph.initializer
                         if param.name in optimized_grad]
    elif args.freeze_net:
        requires_grad = [param.name for param in onnx_model.graph.initializer
                         if "extras" not in param.name
                         and "base_net" not in param.name
                         and "running_mean" not in param.name
                         and "running_var" not in param.name
                         ]
    elif args.freeze_basenet:
        requires_grad = [param.name for param in onnx_model.graph.initializer
                         if "base_net" not in param.name
                         and "running_mean" not in param.name
                         and "running_var" not in param.name
                         ]
    else:
        requires_grad = [param.name for param in onnx_model.graph.initializer
                         if "running_mean" not in param.name
                         and "running_var" not in param.name]

    frozen_params = [param.name for param in onnx_model.graph.initializer if param.name not in requires_grad]

    # Generate the training artifacts.
    artifacts.generate_artifacts(
        onnx_model,
        loss=None,
        optimizer=onnxblock.optim.AdamW(weight_decay=5e-4, eps=1e-8),
        requires_grad=requires_grad,
        frozen_params=frozen_params,
        artifact_directory=args.artifacts_dir_path)

if __name__ == '__main__':
    main()