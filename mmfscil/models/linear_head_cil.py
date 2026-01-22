import torch
import torch.nn.functional as F

from mmcls.models.builder import HEADS
from mmcls.models.heads import LinearClsHead


@HEADS.register_module()
class LinearClsHeadCIL(LinearClsHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 eval_classes=None,
                 metric_type='linear',
                 scale=1.0,
                 *args,
                 **kwargs):
        super().__init__(num_classes=num_classes, in_channels=in_channels, *args, **kwargs)
        self.eval_classes = int(eval_classes) if eval_classes is not None else int(num_classes)
        self.metric_type = str(metric_type).lower()
        self.scale = float(scale)
        if self.metric_type not in ['linear', 'cosine', 'euclidean']:
            raise ValueError(f'metric_type={self.metric_type} is not supported')

    def _compute_logits(self, x):
        if self.metric_type == 'linear':
            return self.fc(x)
        weight = self.fc.weight
        if self.metric_type == 'cosine':
            x = F.normalize(x, dim=1)
            weight = F.normalize(weight, dim=1)
            return self.scale * (x @ weight.t())
        x2 = torch.sum(x * x, dim=1, keepdim=True)
        w2 = torch.sum(weight * weight, dim=1, keepdim=True).t()
        dist = x2 + w2 - 2.0 * (x @ weight.t())
        return -self.scale * dist

    def simple_test(self, x, softmax=True, post_process=True):
        x = self.pre_logits(x)
        cls_score = self._compute_logits(x)
        cls_score = cls_score[:, :self.eval_classes]
        if softmax:
            pred = F.softmax(cls_score, dim=1) if cls_score is not None else None
        else:
            pred = cls_score
        if post_process:
            return self.post_process(pred)
        return pred

    def forward_train(self, x, gt_label, **kwargs):
        x = self.pre_logits(x)
        cls_score = self._compute_logits(x)
        losses = self.loss(cls_score, gt_label, **kwargs)
        return losses
