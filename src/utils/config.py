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


import argparse
import os.path as osp
from glob import glob
from elytra.sys_utils import mkdir


def parse_args_function():
    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "--trainsplit",
        type=str,
        default="smalltrain",
        choices=["train", "smalltrain", "minitrain", "tinytrain"],
        help="Amount to subsample training set.",
    )
    parser.add_argument(
        "--valsplit",
        type=str,
        default="smallval",
        choices=["val", "smallval", "minival"],
        help="Amount to subsample validation set.",
    )
    parser.add_argument("--log_every", type=int, default=50, help="log every k steps")
    parser.add_argument(
        "--eval_every_epoch", type=int, default=5, help="Eval every k epochs"
    )
    parser.add_argument("--bbox_3d_size", type=int, default=400, help="depth axis")
    parser.add_argument("--bbox_3d_size_root", type=int, default=400, help="depth axis")
    parser.add_argument(
        "--output_root_hm_shape", type=int, default=64, help="depth axis"
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=16,
        choices=[16, 32],
        help="Precision for training",
    )
    parser.add_argument(
        "--lr_dec_epoch",
        type=int,
        nargs="+",
        default=[],
        help="Learning rate decay epoch.",
    )
    parser.add_argument(
        "--load_from", type=str, default="", help="Load weights from InterHand format"
    )
    parser.add_argument(
        "--load_ckpt", type=str, default="", help="Load checkpoints from PL format"
    )
    parser.add_argument(
        "--resume_ckpt",
        type=str,
        default="",
        help="Resume training from checkpoint and keep logging in the same comet exp",
    )
    parser.add_argument("--min_epoch", type=int, default=100)
    parser.add_argument("--max_epoch", type=int, default=30000000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--lr_dec_factor", type=int, default=10, help="Learning rate decay factor"
    )
    parser.add_argument(
        "--lr_decay", type=float, default=0.1, help="Learning rate decay factor"
    )
    parser.add_argument(
        "--iter_batch",
        type=int,
        default=16,
        help="Number of images per iteration per GPU",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=16)
    parser.add_argument(
        "--trans_test", type=str, choices=["gt", "rootnet"], default="rootnet"
    )
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--gpu_ids",
        type=int,
        nargs="+",
        default=[0],
        help="Identifies the GPU number to use.",
    )
    parser.add_argument("--annot_subset", type=str, default="all", dest="annot_subset")
    parser.add_argument(
        "-f",
        "--fast",
        dest="fast_dev_run",
        help="single batch for development",
        action="store_true",
    )
    parser.add_argument("--mute", help="No logging", action="store_true")
    parser.add_argument("--no_vis", help="Stop visualization", action="store_true")
    parser.add_argument(
        "--print_summary", help="Print model weights summary", action="store_true"
    )
    parser.add_argument(
        "--eval_on",
        type=str,
        default="",
        choices=["val", "test", "minival", "minitest"],
        help="Test mode set to eval on",
    )

    args = parser.parse_args()
    args.num_gpus = len(args.gpu_ids)

    root_dir = osp.join(".")
    data_dir = osp.join(root_dir, "data")

    args.root_dir = root_dir
    args.data_dir = data_dir
    args.experiment = None
    args.seed = 1

    input_img_size = 256
    output_hm_size = 64
    args.input_img_shape = tuple([input_img_size] * 2)
    args.output_hm_shape = tuple([output_hm_size] * 3)

    if args.fast_dev_run:
        args.num_workers = 0
        args.trainsplit = "minitrain"
        args.valsplit = "minival"
        args.eval_every_epoch = 1

    args.acc_grad_steps = int(1.0 * args.batch_size / (args.iter_batch * args.num_gpus))
    assert args.batch_size == args.iter_batch * args.num_gpus * args.acc_grad_steps

    args.joint_num = 21
    args.beta = 1.0

    args.load_ckpt = None if args.load_ckpt == "" else args.load_ckpt
    args.resume_ckpt = None if args.resume_ckpt == "" else args.resume_ckpt
    assert not (args.load_ckpt is not None and args.resume_ckpt is not None)
    if args.load_ckpt is None:
        args.load_ckpt = args.resume_ckpt
    return args


cfg = parse_args_function()

assert (
    cfg.annot_subset
), "Please set proper annotation subset. Select one of all, human_annot, machine_annot"
cfg.anno_path = "./data/InterHand/annotations/"


if cfg.experiment is None and cfg.eval_on not in ["val", "test", "minitest", "minival"]:
    import elytra.exp_utils as exp_utils

    api_key = ""
    project_name = ""
    workspace = "digit"
    cfg = exp_utils.init_experiment(
        api_key=api_key,
        project_name=project_name,
        workspace=workspace,
        mute=cfg.mute,
        args=cfg,
        tag_list=[],
    )

    output_dir = cfg.output_dir
    cfg.model_dir = osp.join(output_dir, "model_dump")
    cfg.vis_dir = osp.join(output_dir, "vis")
    cfg.log_dir = osp.join(output_dir, "log")
    cfg.result_dir = osp.join(output_dir, "result")
    cfg.ana_dir = osp.join(output_dir, "analysis")

    mkdir(cfg.model_dir)
    mkdir(cfg.vis_dir)
    mkdir(cfg.log_dir)
    mkdir(cfg.ana_dir)
    mkdir(cfg.result_dir)
