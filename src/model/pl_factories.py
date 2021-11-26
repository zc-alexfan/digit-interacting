# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


# Copyright (c) ETH Zurich and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Our code started from the original code in https://github.com/facebookresearch/InterHand2.6M; We have made significant modification to the original code in developing DIGIT.
"""


from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from src.dataset.image_dataset import ImageDataset
from src.model.model import get_model_pl
from src.utils.eval_utils import evaluate
from elytra.pl_module import PL
from src.utils.visualize_pl import visualize_all
from elytra.exp_utils import push_images_disk


def fetch_train_dataloader(cfg, split):
    TrainDataset = ImageDataset

    # data load and construct batch generator
    print("Creating train dataset...")
    trainset_loader = TrainDataset(transforms.ToTensor(), "train", split, cfg)
    batch_generator = DataLoader(
        dataset=trainset_loader,
        batch_size=cfg.num_gpus * cfg.iter_batch,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return batch_generator


def fetch_val_dataloader(cfg, test_set, split, shuffle=False):
    print("Creating " + test_set + " dataset...")
    testset_loader = ImageDataset(transforms.ToTensor(), test_set, split, cfg)
    batch_generator = DataLoader(
        dataset=testset_loader,
        batch_size=cfg.num_gpus * cfg.test_batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return batch_generator


class DIGIT_PL(PL):
    def __init__(self, args):
        super().__init__(
            args,
            get_model_pl,
            evaluate,
            visualize_all,
            push_images_disk,
            "mpjpe_all",
            float("inf"),
        )
        self.model.pose_reg.defrost_all()


def fetch_pl_model(args, experiment):
    model = DIGIT_PL(args)
    return model
