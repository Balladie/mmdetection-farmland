_base_ = ['./mask2former_r50_8xb2-lsj-50e_farmland-panoptic.py']

num_things_classes = 12
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes
image_size = (640, 640)
batch_augments = [
    dict(
        type='BatchFixedSizePad',
        size=image_size,
        img_pad_value=0,
        pad_mask=True,
        mask_pad_value=0,
        pad_seg=False)
]
data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32,
    pad_mask=True,
    mask_pad_value=0,
    pad_seg=False,
    batch_augments=batch_augments)
model = dict(
    data_preprocessor=data_preprocessor,
    panoptic_head=dict(
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        loss_cls=dict(class_weight=[1.0] * num_classes + [0.1])),
    panoptic_fusion_head=dict(
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes),
    test_cfg=dict(panoptic_on=False))

# dataset settings
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        to_float32=True,
        backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', prob=0.5),
    # large scale jittering
    dict(
        type='RandomResize',
        scale=image_size,
        ratio_range=(0.1, 2.0),
        resize_type='Resize',
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=image_size,
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5), by_mask=True),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        to_float32=True,
        backend_args={{_base_.backend_args}}),
    dict(type='Resize', scale=image_size, keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

dataset_type = 'FarmlandDataset'
data_root = 'data/farmland/'

train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        type=dataset_type,
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img=''),
        pipeline=train_pipeline))
val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img=''),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

# 기존 config 파일의 val_evaluator를 수정
val_evaluator = [dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val.json',
    metric=['segm'],
    classwise=True,
    # iou_thrs=[0.05, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75],  # IoU 임계값 범위 확장
    iou_thrs=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],  # IoU 임계값 범위 확장
    outfile_prefix='./work_dirs/mask2former/test_results',
    collect_device='cpu',
    format_only=False,
    backend_args={{_base_.backend_args}},
    metric_items=[
        'mAP', 'mAP_50', 'mAP_75',  # 기본 mAP
        'mAP_s', 'mAP_m', 'mAP_l',  # 크기별 mAP
        'AR@100', 'AR@300', 'AR@1000',   # maxDets에 따른 AR
        'AR_s@1000', 'AR_m@1000', 'AR_l@1000'  # 크기별 AR (maxDets=1000)
    ])]  # 기본 메트릭으로 단순화

test_evaluator = val_evaluator

# 가장 좋은 성능이 나왔던 evaluator - 성능인증시 사용되었던 코드
# val_evaluator = dict(
#     _delete_=True,
#     type='CocoMetric',
#     ann_file=data_root + 'annotations/instances_val.json',
#     metric=['segm'],
#     iou_thrs=[0.05],
#     format_only=False,
#     backend_args={{_base_.backend_args}})
# test_evaluator = val_evaluator

################################### JS
# 기존 config 파일에 추가
# default_hooks = dict(
#     timer=dict(type='IterTimerHook'),
#     logger=dict(
#         type='LoggerHook',
#         interval=50,
#         hooks=[
#             dict(type='TextLoggerHook'),
#             dict(type='TensorboardLoggerHook'),
#             dict(type='WandbLoggerHook',
#                  init_kwargs={'project': 'farmland-segmentation'},
#                  interval=50)
#         ])
# )
################################### JS
# 기존 config 파일에 추가
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),  # 기본 로거만 사용
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

# TensorBoard만 사용하도록 설정
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')
