_base_ = [
    '../_base_/models/resnet_etf.py',
    '../_base_/datasets/synthetic_fscil.py',
    '../_base_/schedules/cifar_200e.py',
    '../_base_/default_runtime.py'
]

TRAIN_MODE = 'edge_cloud'

PROTO_MODE = 'SIM'
SIM_CFG = dict(
    sim_path=None,
    sim_format=None,
    sim_eps=1e-6,
    eig_tol=1e-6,
    hungarian=False,
    hungarian_max_classes=200,
    fallback_steps=50,
    fallback_lr=0.1,
)

model = dict(
    neck=dict(type='MLPFFNNeck', in_channels=640, out_channels=512),
    head=dict(
        num_classes=10,
        eval_classes=6,
        in_channels=512,
        with_len=False,
        proto_mode=PROTO_MODE,
        sim_cfg=SIM_CFG,
    ),
)

inc_start = 6
inc_end = 10
inc_step = 2
copy_list = (1, 1)
step_list = (10, 10)
feat_test = False
mean_neck_feat = True
mean_cur_feat = False
finetune_lr = 0.05

edge_cloud = dict(
    num_clients=3,
    global_rounds=2,
    local_epochs=1,
    local_batch_size=16,
    eval_batch_size=16,
    eval_share='features',
    partition=dict(type='iid', ratios=None, seed=0),
    share_backbone=True,
    share_neck=True,
    share_head=True,
    train_backbone=True,
    use_data_size=True,
    use_entropy=False,
    use_stability=True,
    use_drift=True,
    mutual_eval=dict(score='row_mean', enabled=True),
    weighting=dict(strategy='softmax', beta=5.0, cap=0.6),
    reflect=dict(
        enabled=True,
        threshold=0.2,
        steps=10,
        batch_size=16,
        lr=0.01,
        margin=0.1,
        lambda_reflect=1.0,
        neg_topk=1,
        max_memory=512,
    ),
    log_interval=10,
)
