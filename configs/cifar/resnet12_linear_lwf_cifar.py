_base_ = [
    './resnet12_linear_ft_cifar.py'
]

distill = dict(
    enabled=True,
    type='lwf',
    temperature=2.0,
    lambda_kd=1.0,
)
