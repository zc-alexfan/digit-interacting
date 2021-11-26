import pytorch_lightning as pl
from src.model.pl_factories import fetch_pl_model
from src.model.pl_factories import fetch_val_dataloader
import elytra.torch_utils as torch_utils


def demo_model(args):
    pl.seed_everything(args.seed)
    model = fetch_pl_model(args, args.experiment)

    if 'val' in args.eval_on:
        loader = fetch_val_dataloader(args, 'val', args.eval_on, shuffle=True)
    elif 'test' in args.eval_on:
        loader = fetch_val_dataloader(args, 'test', args.eval_on, shuffle=True)
    else:
        assert False, "Invalid loader (%s)" % (args.eval_on)

    model.cuda()
    model.freeze()
    model.eval()

    batches = []
    for batch_idx, batch in enumerate(loader):
        batch = [torch_utils.dict2dev(data, 'cuda:0') for data in batch]
        # only visualize 3 batches in total
        # change to higher if needed
        if batch_idx > 3:
            break
        batches.append(batch)
    num_examples_per_batch = 4
    model.visualize_batches(batches, '', num_examples_per_batch)


if __name__ == "__main__":
    from src.utils.config import cfg
    demo_model(cfg)
