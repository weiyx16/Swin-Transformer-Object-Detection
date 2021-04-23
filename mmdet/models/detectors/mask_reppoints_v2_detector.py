import numpy as np

import torch

from mmdet.core import bbox2result, bbox_mapping_back, multiclass_nms, merge_aug_masks, bbox_flip
from ..builder import DETECTORS
from .single_stage import SingleStageDetector
import torch.nn.functional as F
from mmdet.models.roi_heads.mask_heads.condconv_mask_head import aligned_bilinear
import copy
from PIL import Image 
@DETECTORS.register_module()
class RepPointsV2MaskDetector(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 mask_inbox=False):
        self.mask_inbox = mask_inbox
        super(RepPointsV2MaskDetector, self).__init__(backbone, neck, bbox_head, train_cfg,
                                                test_cfg, pretrained)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_sem_map=None,
                      gt_sem_weights=None,
                      gt_masks=None):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | list[BitmapMasks]) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        masks = []
        for mask in gt_masks:
            mask_tensor = img.new_tensor(mask.masks)
            mask_tensor = F.pad(mask_tensor, pad=(0, img.size(-1)-mask_tensor.size(-1), 0, img.size(-2)-mask_tensor.size(-2)))
            masks.append(mask_tensor)
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore, gt_sem_map, gt_sem_weights, masks)
        return losses
        
    def merge_aug_results(self, aug_bboxes, aug_scores, img_metas):
        """Merge augmented detection bboxes and scores.

        Args:
            aug_bboxes (list[Tensor]): shape (n, 4*#class)
            aug_scores (list[Tensor] or None): shape (n, #class)
            img_shapes (list[Tensor]): shape (3, ).

        Returns:
            tuple: (bboxes, scores)
        """
        recovered_bboxes = []
        for bboxes, img_info in zip(aug_bboxes, img_metas):
            img_shape = img_info[0]['img_shape']
            scale_factor = img_info[0]['scale_factor']
            flip = img_info[0]['flip']
            flip_direction = img_info[0]['flip_direction']
            bboxes = bbox_mapping_back(bboxes, img_shape, scale_factor, flip,
                                       flip_direction)
            recovered_bboxes.append(bboxes)
        bboxes = torch.cat(recovered_bboxes, dim=0)
            
        if aug_scores is None:
            return bboxes
        else:
            scores = torch.cat(aug_scores, dim=0)
            return bboxes, scores

    def mask2result(self, x, det_labels, inst_inds, img_meta, det_bboxes, pred_instances=None, rescale=True, return_score=False):
        resized_im_h, resized_im_w = img_meta['img_shape'][:2]
        ori_h, ori_w = img_meta['ori_shape'][:2]
        if pred_instances is not None:
            pred_instances = pred_instances[inst_inds]
        else:
            pred_instances = self.bbox_head.pred_instances[inst_inds]

        scale_factor = img_meta['scale_factor'] if rescale else [1, 1, 1, 1]
        pred_instances.boxsz = torch.stack((det_bboxes[:, 2] * scale_factor[2] - det_bboxes[:, 0] * scale_factor[0],
            det_bboxes[:, 3] * scale_factor[3] - det_bboxes[:, 1] * scale_factor[1]), axis=-1)
        mask_logits = self.bbox_head.mask_head(x, pred_instances)
        if len(pred_instances) > 0:
            mask_logits = aligned_bilinear(mask_logits, self.bbox_head.mask_head.head.mask_out_stride)
            mask_logits = mask_logits[:, :, :resized_im_h, :resized_im_w]
            mask_logits = F.interpolate(
                mask_logits,
                size=(ori_h, ori_w),
                mode="bilinear", align_corners=False
            ).squeeze(1)
            mask_pred = (mask_logits > 0.5)
            flip = img_meta['flip']
            flip_direction = img_meta['flip_direction']
            if flip:
                if flip_direction == 'horizontal':
                    mask_pred = mask_pred.cpu().numpy()[:, :, ::-1]
                elif flip_direction == 'vertical':
                    mask_pred = mask_pred.cpu().numpy()[:, ::-1, :]
                else:
                    raise ValueError
            else:
               mask_pred = mask_pred.cpu().numpy()
        else:
            mask_pred = torch.zeros((self.bbox_head.num_classes, *img_meta['ori_shape'][:2]), dtype=torch.int)
        cls_segms = [[] for _ in range(self.bbox_head.num_classes)]  # BG is not included in num_classes
        cls_scores =[[] for _ in range(self.bbox_head.num_classes)]

        for i, label in enumerate(det_labels):
            score = det_bboxes[i][-1]
            if self.mask_inbox:
                mask_pred_ = torch.zeros_like(mask_pred[i])
                det_bbox_ = det_bboxes[i, :-1].clone()
                det_bbox_[[0, 1]], det_bbox_[[2, 3]] = det_bbox_[[0, 1]].floor(), det_bbox_[[2, 3]].ceil()
                det_bbox_ = det_bbox_.int()
                mask_pred_[det_bbox_[1]:det_bbox_[3], det_bbox_[0]:det_bbox_[2]] = mask_pred[i][det_bbox_[1]:det_bbox_[3], det_bbox_[0]:det_bbox_[2]]
                cls_segms[label].append(mask_pred_.cpu().numpy())
            else:
                cls_segms[label].append(mask_pred[i].cpu().numpy() if isinstance(mask_pred[i], torch.Tensor) else mask_pred[i])
            cls_scores[label].append(score.cpu().numpy())
        
        if return_score:
            return cls_segms,cls_scores
        return cls_segms

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.
            [We all use True]
        Returns:
            np.ndarray: proposals
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        if not self.bbox_head.mask_head: # detection only
            bbox_results = [
                bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
                for det_bboxes, det_labels in bbox_list
            ]
            return bbox_results[0]
        else:
            bbox_results = [
                bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
                for det_bboxes, det_labels, _ in bbox_list
            ]
            cls_segms = [
                self.mask2result([xl[[i]] for xl in x], det_labels, inst_inds, img_metas[i], det_bboxes) 
                for i, (det_bboxes, det_labels, inst_inds) in enumerate(bbox_list)
            ]
            
            return bbox_results[0], cls_segms[0]

    def aug_test_simple(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[ndarray]: bbox results of each class
        """
        aug_bboxes = []
        aug_scores = []
        aug_inst_inds = []
        last_inds = 0
        last_inds_list = [last_inds]
        pred_instances = []
        for img, img_meta in zip(imgs, img_metas):
            # only one image in the batch
            # recompute feats to save memory
            x = self.extract_feat(img)
            outs = self.bbox_head(x)
            pred_instances.append(copy.deepcopy(self.bbox_head.pred_instances))
            det_bboxes, det_scores, inst_inds = self.bbox_head.get_bboxes(*outs, img_metas=img_meta,
                                            cfg=self.test_cfg, rescale=False, nms=False)[0]
            del outs
            del x
            torch.cuda.empty_cache()
            aug_bboxes.append(det_bboxes)
            aug_scores.append(det_scores)
            inst_inds = inst_inds + last_inds
            last_inds = torch.max(inst_inds).item() + 1
            last_inds_list.append(last_inds)
            aug_inst_inds.append(inst_inds)

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = self.merge_aug_results(
            aug_bboxes, aug_scores, img_metas)
        merged_inst_inds = torch.cat(aug_inst_inds, dim=0)

        det_bboxes, det_labels, inst_inds, keep = multiclass_nms(merged_bboxes, merged_scores,
                                                self.test_cfg.score_thr,
                                                self.test_cfg.nms,
                                                self.test_cfg.max_per_img, inst_inds=merged_inst_inds)

        # inst_inds to source one and split det_bbox to each img aug
        rec_inst_inds = [[] for i in range(len(last_inds_list) - 1)]
        rec_det_bboxes = [[] for i in range(len(last_inds_list) - 1)]
        rec_det_labels = [[] for i in range(len(last_inds_list) - 1)]
        for inds, db, dl in zip(inst_inds, det_bboxes, det_labels):
            for aug_idx, (b_range, e_range) in enumerate(zip(last_inds_list[:-1], last_inds_list[1:])):
                if inds >= b_range and inds < e_range:
                    inds = inds - b_range
                    rec_inst_inds[aug_idx].append(inds)
                    rec_det_bboxes[aug_idx].append(db)
                    rec_det_labels[aug_idx].append(dl)
                    break

        cls_segms = []
        cls_scores = []
        for img, img_meta, inds, db, dl, pred_instance in zip(imgs, img_metas, rec_inst_inds, rec_det_bboxes, rec_det_labels,pred_instances):
            x = self.extract_feat(img)
            if len(inds) > 0:
                # the output of nms bbox have been flipped and rescale to ori img size
                db = torch.stack(db, dim=0)
                if img_meta[0]['flip']:
                    db[:,:4] = bbox_flip(db[:,:4], img_shape = img_meta[0]['img_shape'], direction = img_meta[0]['flip_direction'])
                _cls_segms,_cls_scores = self.mask2result([xl[[0]] for xl in x], torch.stack(dl, dim=0), torch.stack(inds, dim=0), img_meta[0], db, pred_instance,rescale=True,return_score=True)
                cls_segms.append(_cls_segms)
                cls_scores.append(_cls_scores)
            del x
            torch.cuda.empty_cache()

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        # arrange by cls
        final_cls_segms = [[] for _ in range(self.bbox_head.num_classes)]
        final_cls_scores = [[] for _ in range(self.bbox_head.num_classes)]
        for i in range(len(cls_segms)):
            for cls_idx in range(self.bbox_head.num_classes):
                final_cls_scores[cls_idx].extend(cls_scores[i][cls_idx])
                final_cls_segms[cls_idx].extend(cls_segms[i][cls_idx])

        # re rank by bboxes scores     
        for cls_idx in range(self.bbox_head.num_classes):
            score = final_cls_scores[cls_idx]
            if len(score)>0:
                seg = final_cls_segms[cls_idx]
                idx = np.argsort(-1*np.stack(score))
                seg = [seg[i] for i in idx]
                final_cls_segms[cls_idx]=seg
        
        return bbox_results, final_cls_segms

    def aug_test(self, imgs, img_metas, rescale=False):
        return self.aug_test_simple(imgs, img_metas, rescale)