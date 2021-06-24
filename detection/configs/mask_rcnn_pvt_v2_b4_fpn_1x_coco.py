_base_ = [
    '_base_/models/mask_rcnn_r50_fpn.py',
    '_base_/datasets/coco_instance.py',
    '_base_/schedules/schedule_1x.py',
    '_base_/default_runtime.py'
]
# optimizer
model = dict(
    pretrained='pretrained/pvt_v2_b4.pth',
    backbone=dict(
        type='pvt_v2_b4',
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        num_outs=5))
# optimizer
optimizer = dict(type='AdamW', lr=0.0002 / 1.4, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# dataset settings
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1)