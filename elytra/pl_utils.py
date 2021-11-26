import torch
from elytra.ld_utils import ld2dl
import elytra.torch_utils as torch_utils


def push_checkpoint_metric(key, val):
    val = float(val)
    checkpt_metric = torch.FloatTensor([val])
    result = {key: checkpt_metric}
    return result


def avg_losses_cpu(outputs):
    outputs = ld2dl(outputs)
    for key, val in outputs.items():
        val = [v.cpu() for v in val]
        val = torch.cat(val, dim=0).view(-1)
        outputs[key] = val.mean()
    return outputs


def reform_outputs(out_list):
    out_list_dict = ld2dl(out_list)
    outputs = ld2dl(out_list_dict['out_dict'])
    losses = ld2dl(out_list_dict['loss'])

    for k, tensor in outputs.items():
        outputs[k] = torch.cat(tensor)

    for k, tensor in losses.items():
        tensor = [ten.view(-1) for ten in tensor]
        losses[k] = torch.cat(tensor)

    outputs = {k: torch_utils.tensor2np(v) for k, v in outputs.items()}
    loss_dict = {k: v.mean().item() for k, v in losses.items()}
    return outputs, loss_dict
