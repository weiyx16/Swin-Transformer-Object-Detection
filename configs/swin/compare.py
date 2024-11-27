_base_ = [
    '../_base_/models/mask_rcnn_swin_fpn.py',
    '../_base_/datasets/coco_instance_zip_mstrain_512-800.py',
    '../_base_/schedules/schedule_3x.py', '../_base_/default_runtime.py'
]
# norm_cfg = dict(type='GN', num_groups=16, requires_grad=True)
model = dict(
    backbone=dict(
        type="SwinTransformerV2AltWin",
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_sizes=[[32], [32], [32], [32]],
        pretrained_window_sizes=[[16], [16], [16], [8]],
        cpb_sigmoid=True,
        cpb_scale=8.0,
        global_blocks=[[-1], [-1], [1, 7, 13], [1]],
        use_shift=True,

        ape=False,
        drop_path_rate=0.2,
        patch_norm=True,
        use_checkpoint=False,

    ),
    neck=dict(in_channels=[96, 192, 384, 768]))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'logit_scale': dict(decay_mult=0.),
                                                 'rpe_mlp': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
runner = dict(type='EpochBasedRunner', max_epochs=36)

fp16 = dict(loss_scale='dynamic')