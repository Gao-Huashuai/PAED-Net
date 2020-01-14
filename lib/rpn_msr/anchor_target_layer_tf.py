# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import os
import yaml
import numpy as np
import numpy.random as npr

from .generate_anchors import generate_anchors
from ..utils.cython_bbox import bbox_overlaps, bbox_intersections

# TODO: make fast_rcnn irrelevant
# >>>> obsolete, because it depends on sth outside of this project
from ..fast_rcnn.config import cfg
from ..fast_rcnn.bbox_transform import bbox_transform

# <<<< obsolete

DEBUG = False


def anchor_target_layer(rpn_cls_score, gt_boxes, gt_ishard, dontcare_areas, im_info, _feat_stride=[8,],
                        anchor_scales=[4 ,8, 16, 32], aspect_ratio=[0.5, 1, 2], anchor_thresh=8):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    Parameters
    ----------
    rpn_cls_score: (1, H, W, Ax2) bg/fg scores of previous conv layer
    gt_boxes: (G, 5) vstack of [x1, y1, x2, y2, class]
    gt_ishard: (G, 1), 1 or 0 indicates difficult or not
    dontcare_areas: (D, 4), some areas may contains small objs but no labelling. D may be 0
    im_info: a list of [image_height, image_width, scale_ratios]
    _feat_stride: the downsampling ratio of feature map to the original input image
    anchor_scales: the scales to the basic_anchor (basic anchor is [16, 16])
    ----------
    Returns
    ----------
    rpn_labels : (HxWxA, 1), for each anchor, 0 denotes bg, 1 fg, -1 dontcare
    rpn_bbox_targets: (HxWxA, 4), distances of the anchors to the gt_boxes(may contains some transform)
                            that are the regression objectives
    rpn_bbox_inside_weights: (HxWxA, 4) weights of each boxes, mainly accepts hyper param in cfg
    rpn_bbox_outside_weights: (HxWxA, 4) used to balance the fg/bg,
                            beacuse the numbers of bgs and fgs mays significiantly different
    """
    _anchors = generate_anchors(scales=np.array(anchor_scales), ratios=np.array(aspect_ratio))
    _num_anchors = _anchors.shape[0]

    if DEBUG:
        print 'anchors:'
        print _anchors
        print 'anchor shapes:'
        print np.hstack((
            _anchors[:, 2::4] - _anchors[:, 0::4],
            _anchors[:, 3::4] - _anchors[:, 1::4],
        ))
        _counts = cfg.EPS
        _sums = np.zeros((1, 4))
        _squared_sums = np.zeros((1, 4))
        _fg_sum = 0
        _bg_sum = 0
        _count = 0

    # allow boxes to sit over the edge by a small amount
    _allowed_border =  0
    # map of shape (..., H, W)
    #height, width = rpn_cls_score.shape[1:3]

    im_info = im_info[0]

    # Algorithm:
    #
    # for each (H, W) location i
    #   generate 9 anchor boxes centered on cell i
    #   apply predicted bbox deltas at cell i to each of the 9 anchors
    # filter out-of-image anchors
    # measure GT overlap

    assert rpn_cls_score.shape[0] == 1, \
        'Only single item batches are supported'

    # map of shape (..., H, W)
    height, width = rpn_cls_score.shape[1:3]

    if DEBUG:
        print 'AnchorTargetLayer: height', height, 'width', width
        print ''
        print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
        print 'scale: {}'.format(im_info[2])
        print 'height, width: ({}, {})'.format(height, width)
        print 'rpn: gt_boxes.shape', gt_boxes.shape
        print 'rpn: gt_boxes', gt_boxes

    # 1. Generate proposals from bbox deltas and shifted anchors
    shift_x = np.arange(0, width) * _feat_stride
    shift_y = np.arange(0, height) * _feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y) # in W H order
    # K is H x W
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = _num_anchors
    K = shifts.shape[0]
    all_anchors = (_anchors.reshape((1, A, 4)) +
                   shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))
    total_anchors = int(K * A)

    # only keep anchors inside the image
    inds_inside = np.where(
        (all_anchors[:, 0] >= -_allowed_border) &
        (all_anchors[:, 1] >= -_allowed_border) &
        (all_anchors[:, 2] < im_info[1] + _allowed_border) &  # width
        (all_anchors[:, 3] < im_info[0] + _allowed_border)    # height
    )[0]

    if DEBUG:
        print 'total_anchors', total_anchors
        print 'inds_inside', len(inds_inside)

    # keep only inside anchors
    anchors = all_anchors[inds_inside, :]
    if DEBUG:
        print 'anchors.shape', anchors.shape

    # label: 1 is positive, 0 is negative, -1 is dont care
    # (A)
    labels = np.empty((len(inds_inside), ), dtype=np.float32)
    labels.fill(-1)

    # overlaps between the anchors and the gt boxes
    # overlaps (ex, gt), shape is A x G
    overlaps = bbox_overlaps(
        np.ascontiguousarray(anchors, dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float))
    argmax_overlaps = overlaps.argmax(axis=1) # (A)
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
    gt_argmax_overlaps = overlaps.argmax(axis=0) # G
    gt_max_overlaps = overlaps[gt_argmax_overlaps,
                               np.arange(overlaps.shape[1])]
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

    if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        # assign bg labels first so that positive labels can clobber them
        labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

    # fg label: for each gt, anchor with highest overlap
    labels[gt_argmax_overlaps] = 1
    # fg label: above threshold IOU
    labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

    if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        # assign bg labels last so that negative labels can clobber positives
        labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

    # preclude dontcare areas
    if dontcare_areas is not None and dontcare_areas.shape[0] > 0:
        # intersec shape is D x A
        intersecs = bbox_intersections(
            np.ascontiguousarray(dontcare_areas, dtype=np.float), # D x 4
            np.ascontiguousarray(anchors, dtype=np.float) # A x 4
        )
        intersecs_ = intersecs.sum(axis=0) # A x 1
        labels[intersecs_ > cfg.TRAIN.DONTCARE_AREA_INTERSECTION_HI] = -1

    # preclude hard samples that are highly occlusioned, truncated or difficult to see
    if cfg.TRAIN.PRECLUDE_HARD_SAMPLES and gt_ishard is not None and gt_ishard.shape[0] > 0:
        assert gt_ishard.shape[0] == gt_boxes.shape[0]
        gt_ishard = gt_ishard.astype(int)
        gt_hardboxes = gt_boxes[gt_ishard == 1, :]
        if gt_hardboxes.shape[0] > 0:
            # H x A
            hard_overlaps = bbox_overlaps(
                np.ascontiguousarray(gt_hardboxes, dtype=np.float), # H x 4
                np.ascontiguousarray(anchors, dtype=np.float)) # A x 4
            hard_max_overlaps = hard_overlaps.max(axis=0)  # (A)
            labels[hard_max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = -1
            max_intersec_label_inds = hard_overlaps.argmax(axis=1) # H x 1
            labels[max_intersec_label_inds] = -1 #

    # the gt anchors that are lower than anchor_thresh are reassigned
    fg_inds = np.where(labels == 1)[0]
    filtered_gt = argmax_overlaps[fg_inds]
    each_anchor_num = np.array([np.sum(filtered_gt == g) for g in range(gt_boxes.shape[0])])

    for i, n in enumerate(each_anchor_num):
        if n < anchor_thresh:
            anchor_matched = fg_inds[np.where(filtered_gt == i)[0]]
            anchor_not_matched = np.delete(np.arange(len(inds_inside)), anchor_matched)
            gt_overlaps = overlaps[anchor_not_matched, i]
            argsort_gt_overlaps = np.argsort(-gt_overlaps)
            add_anchor = anchor_not_matched[argsort_gt_overlaps[int(n): int(n + anchor_thresh)]]
            labels[add_anchor] = 1
        else:
            continue

    # subsample positive labels if we have too many
    num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
    fg_inds = np.where(labels == 1)[0]
    # if len(fg_inds) > num_fg:
    #     disable_inds = npr.choice(
    #         fg_inds, size=(len(fg_inds) - num_fg), replace=False)
    #     labels[disable_inds] = -1

    # exclude anchors according the distance
    if len(fg_inds) > num_fg:
        # sorted anchors by the distance between anchors and gt boxes.
        filtered_gt = argmax_overlaps[fg_inds]
        each_anchor_num = np.array([np.sum(filtered_gt == g) for g in range(gt_boxes.shape[0])])
        filtered_anchor_num = np.ceil(1.0 * each_anchor_num / np.sum(each_anchor_num) * (len(fg_inds) - num_fg))
        for i, f in enumerate(filtered_anchor_num):
            anchor_matched = fg_inds[np.where(filtered_gt == i)[0]]
            if anchor_matched is None:
                continue
            distance = anchor_gt_distance(anchors[anchor_matched], gt_boxes[i])
            distance = np.squeeze(distance)
            arg_distance = np.argsort(distance)
            sub_anchor = anchor_matched[arg_distance[0:int(f)]]
            labels[sub_anchor] = -1

    # subsample negative labels if we have too many
    num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(
            bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1
        #print "was %s inds, disabling %s, now %s inds" % (
            #len(bg_inds), len(disable_inds), np.sum(labels == 0))

    bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])

    bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)

    bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
        # uniform weighting of examples (given non-uniform sampling)
        num_examples = np.sum(labels >= 0) + 1
        # positive_weights = np.ones((1, 4)) * 1.0 / num_examples
        # negative_weights = np.ones((1, 4)) * 1.0 / num_examples
        positive_weights = np.ones((1, 4))
        negative_weights = np.zeros((1, 4))
    else:
        assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
        positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                            (np.sum(labels == 1)) + 1)
        negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                            (np.sum(labels == 0)) + 1)
    bbox_outside_weights[labels == 1, :] = positive_weights
    bbox_outside_weights[labels == 0, :] = negative_weights

    if DEBUG:
        _sums += bbox_targets[labels == 1, :].sum(axis=0)
        _squared_sums += (bbox_targets[labels == 1, :] ** 2).sum(axis=0)
        _counts += np.sum(labels == 1)
        means = _sums / _counts
        stds = np.sqrt(_squared_sums / _counts - means ** 2)
        print 'means:'
        print means
        print 'stdevs:'
        print stds

    # map up to original set of anchors
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
    bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

    if DEBUG:
        print 'rpn: max max_overlap', np.max(max_overlaps)
        print 'rpn: num_positive', np.sum(labels == 1)
        print 'rpn: num_negative', np.sum(labels == 0)
        _fg_sum += np.sum(labels == 1)
        _bg_sum += np.sum(labels == 0)
        _count += 1
        print 'rpn: num_positive avg', _fg_sum / _count
        print 'rpn: num_negative avg', _bg_sum / _count

    # labels
    #pdb.set_trace()
    labels = labels.reshape((1, height, width, A))
    rpn_labels = labels

    # bbox_targets
    bbox_targets = bbox_targets \
        .reshape((1, height, width, A * 4))

    rpn_bbox_targets = bbox_targets
    # bbox_inside_weights
    bbox_inside_weights = bbox_inside_weights \
        .reshape((1, height, width, A * 4))
    #assert bbox_inside_weights.shape[2] == height
    #assert bbox_inside_weights.shape[3] == width

    rpn_bbox_inside_weights = bbox_inside_weights

    # bbox_outside_weights
    bbox_outside_weights = bbox_outside_weights \
        .reshape((1, height, width, A * 4))
    #assert bbox_outside_weights.shape[2] == height
    #assert bbox_outside_weights.shape[3] == width

    rpn_bbox_outside_weights = bbox_outside_weights

    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights

def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)


def anchor_gt_distance(anchors, gt_boxes):
    """ Counts the distance of central point between anchors and ground truth boxes. """
    anchors_num = anchors.shape[0]
    gt_num = gt_boxes.shape[0]
    if len(gt_boxes.shape) == 1:
        gt_boxes = gt_boxes.reshape((1, -1))
        gt_num = 1

    anchors_center = np.concatenate([np.array((anchors[:, 2] + anchors[:, 0]) / 2).reshape((-1, 1)),
                                     np.array((anchors[:, 3] + anchors[:, 1]) / 2).reshape((-1, 1))], axis=-1)
    gt_boxes_center = np.concatenate([np.array((gt_boxes[:, 2] + gt_boxes[:, 0]) / 2).reshape((-1, 1)),
                                      np.array((gt_boxes[:, 3] + gt_boxes[:, 1]) / 2).reshape((-1, 1))], axis=-1)
    # uses for loop
    # for a in range(anchors_num):
    #     for g in range(gt_num):
    #         distance[a, g] = math.sqrt((gt_boxes_center[g, 0] - anchors_center[a, 0])**2
    #                                    + (gt_boxes_center[g, 1] - anchors_center[a, 1])**2)
    # uses matrix
    aa = np.tile(anchors_center[:, 0].reshape((-1, 1)), [1, gt_num])
    x_differ = np.square(np.tile(anchors_center[:, 0].reshape((-1, 1)), [1, gt_num]) - gt_boxes_center[:, 0].reshape((1, -1)))
    y_differ = np.square(np.tile(anchors_center[:, 1].reshape((-1, 1)), [1, gt_num]) - gt_boxes_center[:, 1].reshape((1, -1)))
    distance = np.sqrt(x_differ + y_differ)
    return distance
