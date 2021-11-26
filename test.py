import pytorch_lightning as pl
from src.model.pl_factories import fetch_pl_model
from src.model.pl_factories import fetch_val_dataloader
from pytorch_lightning.trainer import Trainer as TrainerPL



def test_model(args):
    pl.seed_everything(args.seed)
    model = fetch_pl_model(args, args.experiment)

    if 'val' in args.eval_on:
        loader = fetch_val_dataloader(args, 'val', args.eval_on)
    elif 'test' in args.eval_on:
        loader = fetch_val_dataloader(args, 'test', args.eval_on)
    else:
        assert False, "Invalid loader (%s)" % (args.eval_on)

    model.cuda()
    model.freeze()
    model.eval()

    trainer = TrainerPL(
            precision=args.precision,
            gpus=args.gpu_ids,
            logger=None,
            progress_bar_refresh_rate=5)
    model.started_training = True
    model.val_set = loader.dataset
    metric_dict = trainer.test(model, loader)
    print(metric_dict)

if __name__ == "__main__":
    from src.utils.config import cfg
    test_model(cfg)
