_base_ = [
    '_base_/models/retinanet_r50_fpn.py',
    '_base_/datasets/coco_detection.py',
    '_base_/schedules/schedule_3x.py',
    '_base_/default_runtime.py'
]
# LastLevelP6P7(256,256)
model = dict(
    pretrained=None,
    backbone=dict(
        type='swin_v2_tiny',
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[192, 384, 768],
        out_channels=256,
        start_level=0,
        # add_extra_convs='on_input',
        add_extra_convs='on_input',
        num_outs=5))
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
