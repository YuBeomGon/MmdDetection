_base_ = [
    '_base_/models/retinanet_r50_fpn.py',
    '_base_/datasets/coco_detection.py',
    '_base_/schedules/schedule_3x.py',
    '_base_/default_runtime.py'
]
# LastLevelP6P7(256,256)
model = dict(
    # pretrained='pretrained/pvt_tiny.pth',
    # pretrained='https://github.com/whai362/PVT/releases/download/v2/pvt_tiny.pth',
    pretrained='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth',
    backbone=dict(
        type='swin_tiny1',
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
