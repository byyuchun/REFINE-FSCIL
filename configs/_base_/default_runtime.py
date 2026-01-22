# checkpoint saving
checkpoint_config = dict(interval=1, max_keep_ckpts=2)
evaluation = dict(interval=1, save_best='auto')
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
    ]
)

dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('train', 1)]

load_from = None
resume_from = None

# Training mode
TRAIN_MODE = 'single'

# Edge-cloud federated training config (active when TRAIN_MODE == 'edge_cloud')
edge_cloud = dict(
    num_clients=2,
    global_rounds=1,
    local_epochs=1,
    local_batch_size=None,
    eval_batch_size=32,
    eval_share='features',  # 'features' or 'images'
    partition=dict(type='iid', ratios=None, classes_per_client=None, seed=0),
    share_backbone=True,
    share_neck=True,
    share_head=True,
    train_backbone=True,
    use_data_size=True,
    use_entropy=True,
    use_stability=True,
    use_drift=True,
    mutual_eval=dict(score='row_mean', enabled=True),  # row_mean or col_mean
    weighting=dict(strategy='softmax', beta=5.0, cap=0.5),
    reflect=dict(
        enabled=True,
        threshold=0.2,
        steps=20,
        batch_size=32,
        lr=0.01,
        margin=0.1,
        lambda_reflect=1.0,
        neg_topk=1,
        max_memory=2048,
    ),
    log_interval=50,
    seed=0,
)

# Distillation/regularization for incremental sessions
distill = dict(
    enabled=False,
    type='lwf',  # lwf or l2
    temperature=2.0,
    lambda_kd=1.0,
    lambda_l2=1.0,
)

# Test configs
mean_neck_feat = True
mean_cur_feat = False
feat_test = False
grad_clip = None
finetune_lr = 0.1
inc_start = 60
inc_end = 100
inc_step = 5

copy_list = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
step_list = (50, 50, 50, 50, 50, 50, 50, 50, 50, 50)
