# Copyright (c) 2025 STMicroelectronics. All rights reserved.
#

import argparse
from onnxruntime.training import artifacts
import onnx
import torch
import os

from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.config import mobilenetv1_ssd_config
from vision.utils.box_utils import generate_ssd_priors


def main():
    parser = argparse.ArgumentParser(
                        description='Script to help with the export of SSDMobileNetV2 model with loss function.')
    parser.add_argument('--weights_path',
                        help='Path of the weights file to use for the basenet.')
    parser.add_argument('--nb_classes',
                        help='Number of classes to be predicted. Default 1.', type=int, default=1)
    parser.add_argument('--img_size', default=300, type=int,
                        help='Desired image size for training. Default = 300.')
    parser.add_argument('--output_model_name',
                        help='Name the exported model. Default = ssd_model_with_outputs_and_loss.', default="ssd_model_with_outputs_and_loss")
    parser.add_argument('--onnx_opset',
                        help='Opset version to use for model export. Default = 17.', type=int, default=17)
    parser.add_argument('--iou_threshold',
                        help='IOU threshold for the loss function. Default = 0.5.', type=int, default=0.5)
    args = parser.parse_args()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    target_filenames = ['mb2-ssd-lite-mp-0_686.pth', 'mb2-imagenet-71_8.pth']
    weights_filename = os.path.basename(args.weights_path)
    if weights_filename not in target_filenames:
        print("Weights filename should be either mb2-ssd-lite-mp-0_686.pth or mb2-imagenet-71_8.pth")
        return False

    img_size = args.img_size

    # Load a model (with pretrained weights)
    ssd_model = create_mobilenetv2_ssd_lite(num_classes=args.nb_classes + 1, onnx_compatible=True)

    class Mbv2SSDnWithLoss(torch.nn.Module):
        def __init__(self, ssd_model, train_mode=True):
            super(Mbv2SSDnWithLoss, self).__init__()
            self.model = ssd_model
            self.model.train(train_mode)
            self.criterion = MultiboxLoss(generate_ssd_priors(mobilenetv1_ssd_config.specs, img_size), iou_threshold=args.iou_threshold, neg_pos_ratio=3,
                                          center_variance=0.1, size_variance=0.2, device=device, export_to_onnx=True)

            # Initialize the base net with pretrained weights
            if weights_filename == target_filenames[0]:
                self.model.init_from_pretrained_ssd(args.weights_path)
            else:
                self.model.init_from_base_net(args.weights_path)

        def forward(self, batch, labels, in_boxes):
            confs, out_boxes = self.model(batch)
            regression_loss, classification_loss = self.criterion(
                confs, out_boxes, labels, in_boxes)
            loss = regression_loss + classification_loss
            # Loss should be the first output in order to correctly generate the artifacts
            return loss, confs, out_boxes

    ssd_model_with_loss = Mbv2SSDnWithLoss(ssd_model, train_mode=True)

    # Generate anchors
    anchors = generate_ssd_priors(mobilenetv1_ssd_config.specs, img_size)
    nb_anchors = anchors.size()[0]

    # The number of anchors depends on the model architecture and the image size
    # Set training option to torch.onnx.TrainingMode.TRAINING to export the model in training friendly mode
    torch.onnx.export(ssd_model_with_loss, (torch.randn(3, 3, img_size, img_size), torch.zeros((3, nb_anchors)), torch.rand(3, nb_anchors, 4)),
                      f"{args.output_model_name}.onnx",
                      input_names=["images", "labels", "in_boxes"], output_names=["loss", "confs", "out_boxes"],
                      dynamic_axes={"images": {0: "batch"},
                                    "labels": {0: "batch", 1: "priors"},
                                    "in_boxes": {0: "batch", 1: "priors", 2: "coordinates"},
                                    "out_boxes": {0: "batch", 1: "priors", 2: "coordinates"},
                                    "confs": {0: "batch", 1: "priors", 2: "confidences"},
                                    "loss": {0: "batch"}}, training=torch.onnx.TrainingMode.TRAINING,
                                    do_constant_folding=False, export_params=True, opset_version=args.onnx_opset)

if __name__ == '__main__':
    main()