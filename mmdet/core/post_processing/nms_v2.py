def multiclass_nms_rpd(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None,
                   inst_inds=None):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int, optional): if there are more than max_num bboxes after
            NMS, only top max_num will be kept. Default to -1.
        score_factors (Tensor, optional): The factors multiplied to scores
            before applying NMS. Default to None.

    Returns:
        tuple: (bboxes, labels, indices), tensors of shape (k, 5),
            (k), and (k). Labels are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
        if inst_inds is not None:
            inst_inds = inst_inds.view(inst_inds.size(0), -1)
    else:
        bboxes = multi_bboxes[:, None].expand(-1, num_classes, 4)
        if inst_inds is not None:
            inst_inds = inst_inds[:, None].expand(-1, num_classes)
    scores = multi_scores[:, :-1]

    # filter out boxes with low scores
    if not torch.onnx.is_in_onnx_export():
        # NonZero not supported  in TensorRT
        # remove low scoring boxes
        valid_mask = scores > score_thr
    bboxes = bboxes[valid_mask]
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = scores[valid_mask]
    labels = valid_mask.nonzero()[:, 1]
    if inst_inds is not None:
        inst_inds = inst_inds[valid_mask]


    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)
        if inst_inds is not None:
            return bboxes, labels, inst_inds, None
        else:
            return bboxes, labels, None

    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    if inst_inds is not None:
        return dets, labels[keep], inst_inds[keep], keep
    else:
        return dets, labels[keep], keep