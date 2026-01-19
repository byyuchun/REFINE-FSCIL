import copy
from typing import List, Dict, Mapping, Optional, Tuple

import numpy as np
from torch.utils.data import Dataset

from mmcls.datasets.builder import DATASETS
from mmcls.datasets.pipelines import Compose


@DATASETS.register_module()
class SyntheticFSCILDataset(Dataset):
    """Synthetic dataset for FSCIL smoke tests seen as random images."""

    def __init__(self,
                 pipeline: List[Dict],
                 num_cls: int = 10,
                 samples_per_class: int = 20,
                 img_size: int = 32,
                 channels: int = 3,
                 subset: str = 'train',
                 few_cls: Optional[Tuple] = None,
                 seed: int = 0,
                 test_mode: bool = False):
        self.pipeline = Compose(pipeline)
        self.num_cls = int(num_cls)
        self.samples_per_class = int(samples_per_class)
        self.img_size = int(img_size)
        self.channels = int(channels)
        self.subset = subset

        if few_cls is not None:
            class_ids = list(few_cls)
        else:
            class_ids = list(range(self.num_cls))
        self.class_ids = class_ids
        self.CLASSES = ['cls_{:03d}'.format(i) for i in range(self.num_cls)]

        self.rng = np.random.RandomState(int(seed) + (0 if subset == 'train' else 1))
        self.data_infos = self._build_infos()

    @property
    def class_to_idx(self) -> Mapping:
        return {_class: i for i, _class in enumerate(self.CLASSES)}

    def _build_infos(self) -> List[Dict]:
        infos = []
        for cls_id in self.class_ids:
            for img_id in range(self.samples_per_class):
                img = self.rng.randint(
                    0, 256,
                    size=(self.img_size, self.img_size, self.channels),
                    dtype=np.uint8)
                info = {
                    'img': img,
                    'gt_label': np.array(cls_id, dtype=np.int64),
                    'cls_id': cls_id,
                    'img_id': img_id,
                }
                infos.append(info)
        return infos

    def __len__(self) -> int:
        return len(self.data_infos)

    def __getitem__(self, idx: int) -> Dict:
        results = copy.deepcopy(self.data_infos[idx])
        return self.pipeline(results)
