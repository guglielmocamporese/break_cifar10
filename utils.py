##################################################
# Imports
##################################################

import pytorch_lightning as pl
import os
import math


def cos_anneal(e0, e1, t0, t1, e):
    """ ramp from (e0, t0) -> (e1, t1) through a cosine schedule based on e \in [e0, e1] """
    alpha = max(0, min(1, (e - e0) / (e1 - e0))) # what fraction of the way through are we
    alpha = 1.0 - math.cos(alpha * math.pi / 2) # warp through cosine
    t = alpha * t1 + (1 - alpha) * t0 # interpolate accordingly
    return t

def linear(e0, e1, t0, t1, e):
    """ linear from (e0, t0) -> (e1, t1) through a cosine schedule based on e \in [e0, e1] """
    alpha = max(0, min(1, (e - e0) / (e1 - e0))) # what fraction of the way through are we
    t = alpha * t1 + (1 - alpha) * t0 # interpolate accordingly
    return t

def cos_anneal_warmup(e0, e1, t0, t1, e_w, e):
    if e >= e_w:
        t = cos_anneal(e0, e1, t0, t1, e)
    else:
        t = linear(e0, e_w, 0, t0, e)
    return t

class DecayLR(pl.Callback):
    def __init__(self, lr_init=3e-4, lr_end=1.25e-6, log_lr=False):
        super(DecayLR, self).__init__()
        self.lr_init = lr_init
        self.lr_end = lr_end
        self.log_lr = log_lr

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        # The step size is annealed from 1e10−4 to 1.25e10−6 over 1,200,000 updates. I use 3e-4
        #t = cos_anneal(0, trainer.max_steps, self.lr_init, self.lr_end, trainer.global_step)
        t = cos_anneal_warmup(0, trainer.max_steps, self.lr_init, self.lr_end, trainer.max_steps // 10, trainer.global_step)
        for g in pl_module.optimizer.param_groups:
            g['lr'] = t

        if self.log_lr:
            pl_module.log('lr', t)

def get_callbacks(args):
    """
    Callbacks for the PyTorch Lightning Trainer.
    """

    # Model checkpoint
    model_checkpoint_clbk = pl.callbacks.model_checkpoint.ModelCheckpoint(
        dirpath=None,
        filename='best',
        monitor=args.metric_monitor,
        save_last=True,
        mode='max',
    )
    model_checkpoint_clbk.CHECKPOINT_NAME_LAST = '{epoch}-{step}'
    callbacks = [
        model_checkpoint_clbk,
        DecayLR(lr_init=args.lr),
    ]
    return callbacks

def get_logger(args):
    """
    Logger for the PyTorchLightning Trainer.
    """
    logger_kind = 'tensorboard' if 'logger' not in args.__dict__ else args.logger
    if logger_kind == 'tensorboard':
        logger = pl.loggers.tensorboard.TensorBoardLogger(
            save_dir=os.path.join(os.getcwd(), 'tmp'),
            name=args.dataset,
        )

    elif logger_kind == 'wandb':
        logger = pl.loggers.WandbLogger(
            save_dir=os.path.join(os.getcwd(), 'tmp'),
            name=args.backbone,
        )

    else:
        raise Exception(f'Error. Logger "{lokker_kind}" is not supported.')
    return logger
