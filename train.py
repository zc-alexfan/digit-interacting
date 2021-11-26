import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer as TrainerPL
from pytorch_lightning.callbacks import ModelCheckpoint
from src.model.pl_factories import fetch_train_dataloader, fetch_pl_model
from src.model.pl_factories import fetch_val_dataloader

def fetch_trainer(args):
    early_stop_cb = None
    checkpoint_cb = ModelCheckpoint(
        filepath=args.model_dir,
        monitor='mpjpe_all',
        save_top_k=-1,
        verbose=False,
        mode='min',
        save_last=True
    )

    weights_summary = 'full' if args.print_summary else 'top'
    backend = 'dp' if args.num_gpus > 1 else None
    trainer = TrainerPL(
            distributed_backend=backend,
            resume_from_checkpoint=args.load_ckpt,
            early_stop_callback=early_stop_cb,
            accumulate_grad_batches=args.acc_grad_steps,
            checkpoint_callback=checkpoint_cb,
            precision=args.precision,
            gpus=args.gpu_ids,
            check_val_every_n_epoch=args.eval_every_epoch,
            max_epochs=args.max_epoch, min_epochs=args.min_epoch,
            progress_bar_refresh_rate=5,
            logger=None,
            weights_summary=weights_summary)
    return trainer


def run_exp(args):

    pl.seed_everything(args.seed)
    train_loader = fetch_train_dataloader(args, args.trainsplit)
    val_loader = fetch_val_dataloader(args, 'val', args.valsplit)
    args.joint_type = train_loader.dataset.joint_type
    args.root_joint_idx = train_loader.dataset.root_joint_idx

    pl.seed_everything(args.seed)
    model = fetch_pl_model(args, args.experiment)
    model.train_set = train_loader.dataset
    model.val_set = val_loader.dataset
    trainer = fetch_trainer(args)

    pl.seed_everything(args.seed)
    trainer.fit(model, train_loader, [val_loader])


if __name__ == "__main__":
    from src.utils.config import cfg
    run_exp(cfg)

