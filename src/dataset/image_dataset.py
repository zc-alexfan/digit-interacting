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


import pickle as pkl
import os.path as osp
from tqdm import tqdm
import torch
import numpy as np
from src.dataset.dataset_utils import downsample, process_anno
from src.utils.preprocessing import load_segm
from src.utils.preprocessing import augmentation
from src.utils.preprocessing import transform_input_to_output_space
import torch.nn.functional as F
from src.dataset.dataset_utils import swap_lr_labels_segm_target_channels
from torch.utils.data import Dataset
import elytra.sys_utils as sys_utils
from src.utils.preprocessing import load_skeleton
import json


def _load_img(img_path, folder_name, txn):
    key = img_path.replace("./data/InterHand/images/", "")
    img = np.array(sys_utils.read_lmdb_image(txn, key))
    return img


def _load_segm(img_path, folder_name, txn_segm):
    segm_key = img_path.replace("./data/%s/images/" % (folder_name), "").replace(
        "jpg", "png"
    )
    img_segm = load_segm(segm_key, txn_segm)
    return img_segm


class ImageDataset(Dataset):
    def __init__(self, transform, mode, split, cfg):
        super().__init__()
        # basic configs and file loading
        self.cfg = cfg
        self.mode = mode  # train, test, val
        folder_name = "InterHand"
        self.folder_name = folder_name
        self.data_path = "./data/%s" % (folder_name)
        self.img_path = "./data/%s/images" % (folder_name)
        self.annot_path = "./data/%s/annotations" % (folder_name)

        if self.mode in ["val", "test"]:
            self.rootnet_output_path = (
                "./data/%s/rootnet_output/rootnet_interhand2.6m_output_%s.json"
                % (folder_name, self.mode)
            )
            print("Loading annotation from {}".format(self.rootnet_output_path))

        self.transform = transform
        self.joint_num = 21  # single hand
        self.root_joint_idx = {"right": 20, "left": 41}
        self.joint_type = {
            "right": np.arange(0, self.joint_num),
            "left": np.arange(self.joint_num, self.joint_num * 2),
        }
        self.skeleton = load_skeleton(
            osp.join(self.annot_path, "skeleton.txt"), self.joint_num * 2
        )

        anno_folder = osp.join(self.annot_path, self.mode)
        self.anno_folder = anno_folder
        print("Load annotation from  " + anno_folder)

        with open(
            osp.join(anno_folder, "InterHand2.6M_" + self.mode + "_camera.json")
        ) as f:
            self.cameras = json.load(f)

        with open(
            osp.join(anno_folder, "InterHand2.6M_" + self.mode + "_joint_3d.json")
        ) as f:
            self.joints = json.load(f)

        with open(
            osp.join(self.data_path, "cache/meta_dict_%s.pkl" % (mode)), "rb"
        ) as f:
            self.fitting_err = pkl.load(f)

        # get bbox and depth
        if (self.mode == "val" or self.mode == "test") and cfg.trans_test == "rootnet":
            print("Get bbox and root depth from " + self.rootnet_output_path)
            self.rootnet_result = {}
            with open(self.rootnet_output_path) as f:
                annot = json.load(f)
            for i in range(len(annot)):
                self.rootnet_result[str(annot[i]["annot_id"])] = annot[i]
        else:
            print("Get bbox and root depth from groundtruth annotation")

        print("Reading pickle..")
        with open(
            osp.join(self.anno_folder, "InterHand2.6M_" + self.mode + "_data.pkl"),
            "rb",
        ) as f:
            self.raw_data = pkl.load(f)
        print("Reading pickle: Done")
        self.txn = sys_utils.fetch_lmdb_reader(
            "data/%s/images/interhand.lmdb" % (self.folder_name)
        )
        self.txn_segm = sys_utils.fetch_lmdb_reader(
            "data/%s/segm_32.lmdb" % (self.folder_name)
        )
        print("Fetched lmdb handle")
        self.raw_data = downsample(self.raw_data, split)

        print("Processing annotation ..")
        raw_data_keys = list(self.raw_data.keys())

        self.datalist = []
        self.datalist_sh = []
        self.datalist_ih = []
        self.sequence_names = []
        for aid in tqdm(raw_data_keys):
            img = self.raw_data[aid]

            # camera stuff
            im_path = img["file_name"]
            hand_type = img["anno"]["hand_type"]

            capture_id = im_path.split("/")[0].replace("Capture", "")
            cam_id = im_path.split("/")[2].replace("cam", "")
            focal = self.cameras[capture_id]["focal"][cam_id]
            princpt = self.cameras[capture_id]["princpt"][cam_id]
            campos = self.cameras[capture_id]["campos"][cam_id]
            camrot = self.cameras[capture_id]["camrot"][cam_id]

            if (mode == "val" or mode == "test") and cfg.trans_test == "rootnet":
                bbox_rootnet = np.array(
                    self.rootnet_result[str(aid)]["bbox"], dtype=np.float32
                )
                abs_depth_rootnet = {
                    "right": self.rootnet_result[str(aid)]["abs_depth"][0],
                    "left": self.rootnet_result[str(aid)]["abs_depth"][1],
                }
            else:
                bbox_rootnet = None
                abs_depth_rootnet = None

            data = process_anno(
                img,
                self.cameras,
                bbox_rootnet,
                abs_depth_rootnet,
                self.joints,
                cfg.trans_test,
                cfg.input_img_shape,
                self.joint_num,
                self.joint_type,
                self.root_joint_idx,
                self.img_path,
                self.mode,
            )

            data["campos"] = np.array(campos)
            data["camrot"] = np.array(camrot)
            data["focal"] = np.array(focal)
            data["princpt"] = np.array(princpt)

            hand_type = data["hand_type"]
            seq_name = data["seq_name"]

            fit_err = self.fitting_err[
                data["img_path"].replace("./data/%s/images/" % (self.folder_name), "")
            ]
            right_valid = int(fit_err["right_fit_err"] is not None)
            left_valid = int(fit_err["left_fit_err"] is not None)

            if hand_type == "interacting":
                mano_valid = 1 if left_valid and right_valid else 0
            elif hand_type == "left":
                mano_valid = 1 if left_valid and not right_valid else 0
            elif hand_type == "right":
                mano_valid = 1 if not left_valid and right_valid else 0
            else:
                assert False
            data["mano_valid"] = mano_valid

            if hand_type == "right" or hand_type == "left":
                self.datalist_sh.append(data)
            else:
                self.datalist_ih.append(data)
            if seq_name not in self.sequence_names:
                self.sequence_names.append(seq_name)

        self.datalist = self.datalist_sh + self.datalist_ih
        print(
            "Number of annotations in single hand sequences: "
            + str(len(self.datalist_sh))
        )
        print(
            "Number of annotations in interacting hand sequences: "
            + str(len(self.datalist_ih))
        )
        print("Total number of annotations: " + str(len(self.datalist)))

    def __getitem__(self, idx):
        cfg = self.cfg
        data = self.datalist[idx]
        img_path = data["img_path"]
        bbox = data["bbox"]
        joint = data["joint"]
        hand_type = data["hand_type"]
        hand_type_valid = data["hand_type_valid"]
        mano_valid = data["mano_valid"]
        joint_cam = joint["cam_coord"].copy()
        joint_img = joint["img_coord"].copy()
        joint_valid = joint["valid"].copy()
        hand_type = self.handtype_str2array(hand_type)

        # joint coord in cam space
        joint_coord = np.concatenate((joint_img, joint_cam[:, 2, None]), 1)

        # image and segm load
        img = _load_img(img_path, self.folder_name, self.txn)
        img_segm = _load_segm(img_path, self.folder_name, self.txn_segm)

        hw_size = img.shape[:2]

        segm_valid = mano_valid
        img_segm = (
            np.stack((img_segm, img_segm, img_segm), axis=2)
            if img_segm is not None
            else None
        )

        # augmentation
        (
            img,
            img_segm,
            joint_coord,
            joint_valid,
            hand_type,
            _,
            inv_trans,
            do_flip,
        ) = augmentation(
            img,
            img_segm,
            bbox,
            joint_coord,
            joint_valid,
            hand_type,
            self.mode,
            self.joint_type,
            cfg.input_img_shape,
        )

        if int(segm_valid) == 1:
            img_segm = torch.FloatTensor(img_segm)
            img_segm = img_segm.permute(2, 0, 1)

            if do_flip:
                img_segm = swap_lr_labels_segm_target_channels(img_segm)
            img_segm_256 = img_segm.clone().long()
        else:
            # placeholder segm
            # will be flagged out in the loss with segm_valid
            img_segm_256 = torch.zeros(3, 256, 256).long()

        # downsample to target resolution
        img_segm_256 = (
            F.interpolate(img_segm_256[None, :, :, :].float(), 128, mode="nearest")
            .long()
            .squeeze()
        )

        # relative root depth in cam space
        rel_root_depth = np.array(
            [
                joint_coord[self.root_joint_idx["left"], 2]
                - joint_coord[self.root_joint_idx["right"], 2]
            ],
            dtype=np.float32,
        ).reshape(1)
        root_valid = (
            np.array(
                [
                    joint_valid[self.root_joint_idx["right"]]
                    * joint_valid[self.root_joint_idx["left"]]
                ],
                dtype=np.float32,
            ).reshape(1)
            if hand_type[0] * hand_type[1] == 1
            else np.zeros((1), dtype=np.float32)
        )

        # transform joints in cam space (nonzero root) to heatmap scale
        # same transform is done for: relative root depth
        (
            joint_coord,
            joint_valid,
            rel_root_depth,
            root_valid,
        ) = transform_input_to_output_space(
            joint_coord,
            joint_valid,
            rel_root_depth,
            root_valid,
            self.root_joint_idx,
            self.joint_type,
            cfg.input_img_shape,
            cfg.output_hm_shape,
            cfg.bbox_3d_size,
            cfg.bbox_3d_size_root,
            cfg.output_root_hm_shape,
        )
        img = self.transform(img.astype(np.float32)) / 255.0

        inputs = {"img": img}
        targets = {
            "joint_coord": joint_coord,
            "rel_root_depth": rel_root_depth,
            "hand_type": hand_type,
            "segm_256": img_segm_256,
            "segm_valid": segm_valid,
        }
        meta_info = {
            "joint_valid": joint_valid,
            "root_valid": root_valid,
            "hand_type_valid": hand_type_valid,
            "inv_trans": inv_trans,
            "capture": int(data["capture"]),
            "cam": int(data["cam"]),
            "frame": int(data["frame"]),
            "idx": idx,
            "im_path": img_path,
            "campos": data["campos"],
            "camrot": data["camrot"],
            "focal": data["focal"],
            "princpt": data["princpt"],
            "abs_depth_left": data["abs_depth"]["left"],
            "abs_depth_right": data["abs_depth"]["right"],
            "hw_size": hw_size,
            "flipped": do_flip,
        }
        return inputs, targets, meta_info

    def handtype_str2array(self, hand_type):
        if hand_type == "right":
            return np.array([1, 0], dtype=np.float32)
        elif hand_type == "left":
            return np.array([0, 1], dtype=np.float32)
        elif hand_type == "interacting":
            return np.array([1, 1], dtype=np.float32)
        else:
            assert 0, print("Not supported hand type: " + hand_type)

    def __len__(self):
        return len(self.datalist)
