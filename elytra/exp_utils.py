from comet_ml import Experiment, ExistingExperiment
from pprint import pprint
import socket
import elytra.sys_utils as sys_utils
from tqdm import tqdm
import torch
import numpy as np
import torch
import os.path as op
import time
import uuid


def log_dict(experiment, metric_dict, step, postfix=None):
    if experiment is None:
        return
    for key, value in metric_dict.items():
        if postfix is not None:
            key = key + postfix
        if isinstance(value, torch.Tensor) and len(value.view(-1)) == 1:
            value = value.item()

        if isinstance(value, (int, float, np.float32)):
            experiment.log_metric(key, value, step=step)


class Experiment:
    def __init__(self):
        # unique id
        self.uuid = uuid.uuid4().hex

    def log_parameters(self, params):
        pass

    def get_key(self):
        return self.uuid

    def set_name(self, name):
        pass

    def log_image(self, im_np, im_name):
        print("Log %s" % (im_name))

    def log_metric(self, key, val, step=None):
        pass

    def log_epoch_end(self, epoch):
        pass


def init_experiment(api_key, project_name, workspace, mute, args, tag_list):
    experiment = Experiment()
    args.commit_hash = sys_utils.get_commit_hash()
    experiment.log_parameters(vars(args))
    args.experiment = experiment
    args.exp_key = experiment.get_key()[:9]
    experiment.set_name(args.exp_key)
    args.output_dir = "./logs/%s" % (args.exp_key)
    sys_utils.mkdir_p(args.output_dir)
    pprint(vars(args))
    return args


def push_images(experiment, all_im_list, no_tqdm=False, verbose=True):
    if verbose:
        print("Pushing PIL images")
        tic = time.time()
    iterator = all_im_list if no_tqdm else tqdm(all_im_list)
    for im in iterator:
        if "fig_name" in im.keys():
            experiment.log_image(np.array(im["im"]), im["fig_name"])
        else:
            experiment.log_image(np.array(im["im"]), "unnamed")
    if verbose:
        toc = time.time()
        print("Done pushing PIL images (%.1fs)" % (toc - tic))


def push_images_disk(im_list, global_step):
    out_folder = "visualization"
    for im in im_list:
        im_path = im["fig_name"].replace("./", "").replace("/", "__") + ".png"
        im["im"].save(op.join(out_folder, im_path))
