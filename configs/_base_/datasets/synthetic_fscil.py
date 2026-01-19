img_size = 32
meta_keys = ('cls_id', 'img_id')

train_pipeline = [
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'], meta_keys=meta_keys)
]

test_pipeline = [
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'], meta_keys=meta_keys)
]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=0,
    train_dataloader=dict(
        persistent_workers=False,
    ),
    val_dataloader=dict(
        persistent_workers=False,
    ),
    test_dataloader=dict(
        persistent_workers=False,
    ),
    train=dict(
        type='SyntheticFSCILDataset',
        pipeline=train_pipeline,
        num_cls=10,
        samples_per_class=20,
        subset='train',
        img_size=img_size,
        channels=3,
        seed=0,
    ),
    val=dict(
        type='SyntheticFSCILDataset',
        pipeline=test_pipeline,
        num_cls=10,
        samples_per_class=10,
        subset='test',
        img_size=img_size,
        channels=3,
        seed=1,
    ),
    test=dict(
        type='SyntheticFSCILDataset',
        pipeline=test_pipeline,
        num_cls=10,
        samples_per_class=10,
        subset='test',
        img_size=img_size,
        channels=3,
        seed=1,
    ),
)
