import torch
import pytorch_lightning as pl
import elytra.pl_utils as pl_utils
import torch.optim as optim
from elytra.exp_utils import log_dict
from elytra.pl_utils import push_checkpoint_metric, avg_losses_cpu
import time


class PL(pl.LightningModule):
    def __init__(
            self, args, get_model_pl_fn, evaluate_fn, visualize_all_fn,
            push_images_fn, tracked_metric, metric_init_val):
        super().__init__()
        self.args = args
        self.evaluate = evaluate_fn
        self.tracked_metric = tracked_metric
        self.metric_init_val = metric_init_val

        self.experiment = self.args.experiment
        self.model = get_model_pl_fn(args)
        self.model.load_pretrained(self.args.load_from)
        self.started_training = False
        self.loss_dict_vec = []
        self.has_applied_decay = False
        self.visualize_all = visualize_all_fn
        self.push_images = push_images_fn

    def set_training_flags(self):
        self.started_training = True

    def load_from_ckpt(self, ckpt_path):
        sd = torch.load(ckpt_path)['state_dict']
        print(self.load_state_dict(sd))

    def training_step(self, batch, batch_idx):
        self.set_training_flags()

        inputs, targets, meta_info = batch

        loss = self.model(inputs, targets, meta_info, 'train')
        loss = {k: loss[k].mean().view(-1) for k in loss}
        total_loss = sum(loss[k] for k in loss)

        loss_dict = {'total_loss': total_loss, 'loss': total_loss}
        loss_dict.update(loss)

        log_every = self.args.log_every
        self.loss_dict_vec.append(loss_dict)
        self.loss_dict_vec = self.loss_dict_vec[len(self.loss_dict_vec)-log_every:]
        if batch_idx % log_every == 0 and batch_idx != 0:
            running_loss_dict = avg_losses_cpu(self.loss_dict_vec)
            log_dict(
                self.experiment, running_loss_dict,
                step=self.global_step, postfix='__train')
        return loss_dict

    def training_epoch_end(self, outputs):
        outputs = avg_losses_cpu(outputs)
        self.experiment.log_epoch_end(self.current_epoch)
        return outputs

    def validation_step(self, batch, batch_idx):
        out = self.inference_step(batch, batch_idx)
        return out

    def validation_epoch_end(self, outputs):
        return self.inference_epoch_end(outputs, postfix='__val')

    def test_step(self, batch, batch_idx):
        out = self.inference_step(batch, batch_idx)
        return out

    def test_epoch_end(self, outputs):
        result = self.inference_epoch_end(outputs, postfix='__test')
        return result

    def inference_step(self, batch, batch_idx):
        if self.training:
            self.eval()
        with torch.no_grad():
            inputs, targets, meta_info = batch
            out, loss = self.model(inputs, targets, meta_info, 'test')

            del batch
            return {'out_dict': out, 'loss': loss}

    def inference_epoch_end(self, out_list, postfix):
        if not self.started_training:
            self.started_training = True
            result = push_checkpoint_metric(
                self.tracked_metric, self.metric_init_val)
            return result

        # unpack
        outputs, loss_dict = pl_utils.reform_outputs(out_list)

        # evaluate metrics
        metric_dict, out_dict = self.evaluate(
            self.args, outputs, self.val_set.datalist,
            self.val_set.joint_num, self.val_set.root_joint_idx,
            self.val_set.joint_type, self.val_set.skeleton,
            verbose=False)

        if 'test' in postfix:
            return metric_dict

        loss_metric_dict = {}
        loss_metric_dict.update(metric_dict)
        loss_metric_dict.update(loss_dict)

        log_dict(
            self.experiment, loss_metric_dict,
            step=self.global_step, postfix=postfix)

        result = push_checkpoint_metric(
            self.tracked_metric, loss_metric_dict[self.tracked_metric])
        return result

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, self.args.lr_dec_epoch,
            gamma=self.args.lr_decay)
        return [optimizer], [scheduler]

    def visualize_batches(self, batches, postfix, num_examples,  no_tqdm=True):
        im_list = []
        if self.training:
            self.eval()

        tic = time.time()
        for batch_idx, batch in enumerate(batches):
            with torch.no_grad():
                inputs, targets, meta_info = batch
                vis_dict = self.model(inputs, targets, meta_info, 'vis')
                curr_im_list = self.visualize_all(
                        vis_dict, num_examples,
                        postfix=postfix, no_tqdm=no_tqdm)
                self.push_images(curr_im_list, self.global_step)
                im_list += curr_im_list
                print('Rendering: %d/%d' % (batch_idx + 1, len(batches)))

        print('Done rendering (%.1fs)' % (time.time() - tic))
        return im_list
