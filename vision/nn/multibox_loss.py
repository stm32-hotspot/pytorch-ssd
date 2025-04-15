import torch.nn as nn
import torch.nn.functional as F
import torch


from ..utils import box_utils


class MultiboxLoss(nn.Module):
    def __init__(self, priors, iou_threshold, neg_pos_ratio,
                 center_variance, size_variance, device, export_to_onnx: bool):
        """Implement SSD Multibox Loss.

        Basically, Multibox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(MultiboxLoss, self).__init__()
        self.iou_threshold = iou_threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.priors = priors
        self.priors.to(device)
        self.export_to_onnx = export_to_onnx

    def forward(self, confidence, predicted_locations, labels, gt_locations):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            boxes (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        num_classes = confidence.size(2)
        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
            mask = box_utils.hard_negative_mining(loss, labels, self.neg_pos_ratio)

        confidence = confidence[mask, :]
        # To avoid shape inference error (for SoftmaxCrossEntropy onnx operator, labels can't be float)
        # cf https://onnx.ai/onnx/operators/onnx__SoftmaxCrossEntropyLoss.html
        if self.export_to_onnx:
            labels = labels.to(torch.int32)

        classification_loss = F.cross_entropy(confidence.reshape(-1, num_classes), labels[mask], size_average=False)
        pos_mask = labels > 0
        predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
        gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)

        # Smooth_l1_loss not supported in onnx: replace by mse loss
        """ smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, size_average=False) """
        mse_loss = F.mse_loss(predicted_locations, gt_locations, size_average=False)
        num_pos = gt_locations.size(0)
        return mse_loss/num_pos, classification_loss/num_pos
