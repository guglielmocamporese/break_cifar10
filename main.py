##################################################
# Imports
##################################################

import sys
import pytorch_lightning as pl

# Custom
from config import get_args
from datasets.dataloaders import get_dataloaders
from utils import get_logger, get_callbacks
from models.classifier import get_model
from trainer import get_trainer


def main(args):

    # Dataloaders
    dls = get_dataloaders(args)
    
    # Model
    model = get_model(args)

    # Trainer
    trainer = get_trainer(args, dls)

    # Mode
    if args.mode in ['train', 'training']:
        trainer.fit(model, dls['train_aug'], dls['validation'])
        trainer.validate(model=None, val_dataloaders=dls['validation'])

    elif args.mode in ['validate', 'validation', 'validating']:
        trainer.validate(model, val_dataloaders=dls['validation'])

    else:
        raise Exception(f'Error. Model "{args.mode}" not supported.')


##################################################
# Imports
##################################################

if __name__ == '__main__':

    # Args
    args = get_args(sys.stdin)

    # Set seed
    pl.seed_everything(args.seed)

    # Run main
    main(args)
    
