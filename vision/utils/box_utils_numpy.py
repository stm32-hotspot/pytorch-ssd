import collections

from typing import List
import itertools
import math
import numpy as np

SSDBoxSizes = collections.namedtuple('SSDBoxSizes', ['min', 'max'])

SSDSpec = collections.namedtuple(
    'SSDSpec', ['feature_map_size', 'shrinkage', 'box_sizes', 'aspect_ratios'])


def generate_ssd_priors(specs: List[SSDSpec], image_size, clamp=True):
    """Generate SSD Prior Boxes.

    It returns the center, height and width of the priors. The values are relative to the image size
    Args:
        specs: SSDSpecs about the shapes of sizes of prior boxes. i.e.
            specs = [
                SSDSpec(38, 8, SSDBoxSizes(30, 60), [2]),
                SSDSpec(19, 16, SSDBoxSizes(60, 111), [2, 3]),
                SSDSpec(10, 32, SSDBoxSizes(111, 162), [2, 3]),
                SSDSpec(5, 64, SSDBoxSizes(162, 213), [2, 3]),
                SSDSpec(3, 100, SSDBoxSizes(213, 264), [2]),
                SSDSpec(1, 300, SSDBoxSizes(264, 315), [2])
            ]
        image_size: image size.
        clamp: if true, clamp the values to make fall between [0.0, 1.0]
    Returns:
        priors (num_priors, 4): The prior boxes represented as [[center_x, center_y, w, h]]. All the values
            are relative to the image size.
    """
    priors = []
    for spec in specs:
        scale = image_size / spec.shrinkage
        for j, i in itertools.product(range(spec.feature_map_size), repeat=2):
            x_center = (i + 0.5) / scale
            y_center = (j + 0.5) / scale

            # small sized square box
            size = spec.box_sizes.min
            h = w = size / image_size
            priors.append([
                x_center,
                y_center,
                w,
                h
            ])

            # big sized square box
            size = math.sqrt(spec.box_sizes.max * spec.box_sizes.min)
            h = w = size / image_size
            priors.append([
                x_center,
                y_center,
                w,
                h
            ])

            # change h/w ratio of the small sized box
            size = spec.box_sizes.min
            h = w = size / image_size
            for ratio in spec.aspect_ratios:
                ratio = math.sqrt(ratio)
                priors.append([
                    x_center,
                    y_center,
                    w * ratio,
                    h / ratio
                ])
                priors.append([
                    x_center,
                    y_center,
                    w / ratio,
                    h * ratio
                ])

    priors = np.array(priors, dtype=np.float32)
    if clamp:
        np.clip(priors, 0.0, 1.0, out=priors)
    return priors


def convert_locations_to_boxes(locations, priors, center_variance,
                               size_variance):
    """Convert regressional location results of SSD into boxes in the form of (center_x, center_y, h, w).

    The conversion:
        $$predicted\_center * center_variance = \frac {real\_center - prior\_center} {prior\_hw}$$
        $$exp(predicted\_hw * size_variance) = \frac {real\_hw} {prior\_hw}$$
    We do it in the inverse direction here.
    Args:
        locations (batch_size, num_priors, 4): the regression output of SSD. It will contain the outputs as well.
        priors (num_priors, 4) or (batch_size/1, num_priors, 4): prior boxes.
        center_variance: a float used to change the scale of center.
        size_variance: a float used to change of scale of size.
    Returns:
        boxes:  priors: [[center_x, center_y, h, w]]. All the values
            are relative to the image size.
    """
    # priors can have one dimension less.
    if len(priors.shape) + 1 == len(locations.shape):
        priors = np.expand_dims(priors, 0)
    return np.concatenate([
        locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2],
        np.exp(locations[..., 2:] * size_variance) * priors[..., 2:]
    ], axis=len(locations.shape) - 1)


def convert_boxes_to_locations(center_form_boxes, center_form_priors, center_variance, size_variance):
    # priors can have one dimension less
    if len(center_form_priors.shape) + 1 == len(center_form_boxes.shape):
        center_form_priors = np.expand_dims(center_form_priors, 0)
    return np.concatenate([
        (center_form_boxes[..., :2] - center_form_priors[..., :2]) / center_form_priors[..., 2:] / center_variance,
        np.log(center_form_boxes[..., 2:] / center_form_priors[..., 2:]) / size_variance
    ], axis=len(center_form_boxes.shape) - 1)


def area_of(left_top, right_bottom):
    """Compute the areas of rectangles given two corners.

    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.

    Returns:
        area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.

    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def assign_priors(gt_boxes, gt_labels, corner_form_priors, iou_threshold):
    """Assign ground truth boxes and targets to priors.

    Args:
        gt_boxes (num_targets, 4): ground truth boxes.
        gt_labels (num_targets): labels of targets.
        corner_form_priors (num_priors, 4): corner form priors
        iou_threshold: threshold to assign ground truth to a prior.

    Returns:
        boxes (num_priors, 4): real values for priors.
        labels (num_priors): labels for priors.
    """
    # Calculate the IOU between gt_boxes and corner_form_priors
    ious = iou_of(np.expand_dims(gt_boxes, axis=0),
                  np.expand_dims(corner_form_priors, axis=1))

    # Find the best ground truth for each prior
    best_target_per_prior_index = np.argmax(ious, axis=1)
    best_target_per_prior = ious[np.arange(
        ious.shape[0]), best_target_per_prior_index]

    # Find the best prior for each ground truth
    best_prior_per_target_index = np.argmax(ious, axis=0)
    best_prior_per_target = ious[best_prior_per_target_index, np.arange(
        ious.shape[1])]

    # Ensure every ground truth is assigned to at least one prior
    for target_index, prior_index in enumerate(best_prior_per_target_index):
        best_target_per_prior_index[prior_index] = target_index
    # Using 2 to ensure assignment
    best_target_per_prior[best_prior_per_target_index] = 2.0

    # Set labels for each prior
    labels = gt_labels[best_target_per_prior_index]
    labels[best_target_per_prior < iou_threshold] = 0  # Background class

    # Set boxes for each prior
    boxes = gt_boxes[best_target_per_prior_index]

    return boxes, labels


def center_form_to_corner_form(locations):
    return np.concatenate([locations[..., :2] - locations[..., 2:]/2,
                     locations[..., :2] + locations[..., 2:]/2], len(locations.shape) - 1)


def corner_form_to_center_form(boxes):
    return np.concatenate([
        (boxes[..., :2] + boxes[..., 2:]) / 2,
         boxes[..., 2:] - boxes[..., :2]
    ], len(boxes.shape) - 1)


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """

    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
         picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    #_, indexes = scores.sort(descending=True)
    indexes = np.argsort(scores)
    #indexes = indexes[:candidate_size]
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        #current = indexes[0]
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        #indexes = indexes[1:]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]


# def nms(box_scores, nms_method=None, score_threshold=None, iou_threshold=None,
#         sigma=0.5, top_k=-1, candidate_size=200):
#     if nms_method == "soft":
#         return soft_nms(box_scores, score_threshold, sigma, top_k)
#     else:
#         return hard_nms(box_scores, iou_threshold, top_k, candidate_size=candidate_size)

#
# def soft_nms(box_scores, score_threshold, sigma=0.5, top_k=-1):
#     """Soft NMS implementation.
#
#     References:
#         https://arxiv.org/abs/1704.04503
#         https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/cython_nms.pyx
#
#     Args:
#         box_scores (N, 5): boxes in corner-form and probabilities.
#         score_threshold: boxes with scores less than value are not considered.
#         sigma: the parameter in score re-computation.
#             scores[i] = scores[i] * exp(-(iou_i)^2 / simga)
#         top_k: keep top_k results. If k <= 0, keep all the results.
#     Returns:
#          picked_box_scores (K, 5): results of NMS.
#     """
#     picked_box_scores = []
#     while box_scores.size(0) > 0:
#         max_score_index = torch.argmax(box_scores[:, 4])
#         cur_box_prob = torch.tensor(box_scores[max_score_index, :])
#         picked_box_scores.append(cur_box_prob)
#         if len(picked_box_scores) == top_k > 0 or box_scores.size(0) == 1:
#             break
#         cur_box = cur_box_prob[:-1]
#         box_scores[max_score_index, :] = box_scores[-1, :]
#         box_scores = box_scores[:-1, :]
#         ious = iou_of(cur_box.unsqueeze(0), box_scores[:, :-1])
#         box_scores[:, -1] = box_scores[:, -1] * torch.exp(-(ious * ious) / sigma)
#         box_scores = box_scores[box_scores[:, -1] > score_threshold, :]
#     if len(picked_box_scores) > 0:
#         return torch.stack(picked_box_scores)
#     else:
#         return torch.tensor([])
