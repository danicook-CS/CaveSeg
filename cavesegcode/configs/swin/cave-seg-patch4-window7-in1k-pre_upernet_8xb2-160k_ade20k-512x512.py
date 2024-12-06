_base_ = [
    '/home/afrl/mmsegmentation/configs/_base_/models/upernet_swin.py', '/home/afrl/mmsegmentation/configs/_base_/datasets/ade20k.py',
    '/home/afrl/mmsegmentation/configs/_base_/default_runtime.py', '/home/afrl/mmsegmentation/configs/_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
# checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'  # noqa
# checkpoint_file = '/blue/mdjahiduislam/adnanabdullah/mmsegmentation/work_dirs/cave-seg-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512/iter_160000.pth'  # noqa
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        # init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        embed_dims=48, #was 96
        depths=[2, 2, 4, 2], #was2262
        num_heads=[3, 6, 12, 24], #was 3 to 24
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True),
    decode_head=dict(in_channels=[48, 96, 192, 384], num_classes=13), #was  96 to 768
    auxiliary_head=dict(in_channels=192, num_classes=13)) #was 384

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.01), #was .00006
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader
