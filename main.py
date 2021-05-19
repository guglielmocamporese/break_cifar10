##################################################
# Imports
##################################################

import sys
import pytorch_lightning as pl
import torch
from tqdm import tqdm

# Custom
from config import get_args
from datasets.dataloaders import get_dataloaders
from utils import get_logger, get_callbacks
from models.classifier import get_model
from trainer import get_trainer
from metrics import accuracy


def main(args):

    if args.mode != 'ensemble':

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
        trainer.test(model=None, test_dataloaders=dls['test'])

    elif args.mode in ['validate', 'validation', 'validating']:
        trainer.validate(model, val_dataloaders=dls['validation'])

    elif args.mode in ['test', 'testing']:
        trainer.test(model, test_dataloaders=dls['test'])

    elif args.mode == 'ensemble':
        predictions = []
        labels = []
        backbones = args.backbone
        model_checkpoints = args.model_checkpoint
        for backbone, model_checkpoint in zip(backbones, model_checkpoints):
            args.backbone = backbone
            args.model_checkpoint = model_checkpoint

            dls = get_dataloaders(args)
            model = get_model(args)

            # Predict
            model.eval().cuda()
            preds = []
            lbls = []
            for x, y in tqdm(dls['test']):
                logits = model(x.cuda()).detach().cpu()
                preds += [logits]
                lbls += [y]
            preds = torch.cat(preds, 0)
            predictions += [preds]
            lbls  = torch.cat(lbls, 0)
            labels += [lbls]
            acc_single = accuracy(preds, lbls)
            print(f'{backbone} accuracy: {acc_single}')
        predictions = torch.stack(predictions, 1).mean(1)
        labels = labels[0]
        acc = accuracy(predictions, labels)
        print(f'Ensamble Accuracy: {acc}')

    else:
        raise Exception(f'Error. Model "{args.mode}" not supported.')


##################################################
# Imports
##################################################

if __name__ == '__main__':

    # Args
    args = get_args(sys.stdin, verbose=True)

    # Set seed
    pl.seed_everything(args.seed)

    # Run main
    main(args)
    
