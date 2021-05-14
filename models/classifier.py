##################################################
# Imports
##################################################

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam

# Custom
from models.vit import vit
from models.resnet import resnet
from models.mlp_mixer import mlp_mixer
from losses import cross_entropy
from metrics import accuracy


##################################################
# Classifier
##################################################

class Classifier(pl.LightningModule):
    """
    Lightning module that defines the Jisaw Vision Transformer Model.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_classes = 10
        self.backbone = self._get_backbone()

    def forward_feat(self, x):
        """
        Forward features function of the model.
        """
        x = self.backbone(x)
        return x

    def _get_backbone(self):
        if self.args.backbone == 'vit':
            backbone = vit.vit_small_patch16_224(num_classes=self.num_classes, pretrained=True)

        elif self.args.backbone == 'resnet18':
            backbone = resnet.ResNet18(num_classes=self.num_classes)

        elif self.args.backbone == 'resnet34':
            backbone = resnet.ResNet34(num_classes=self.num_classes)
            
        elif self.args.backbone == 'resnet50':
            backbone = resnet.ResNet50(num_classes=self.num_classes)

        elif self.args.backbone == 'mlp_mixer':
            backbone = mlp_mixer.MLPMixer(image_size=32, patch_size=8, dim=512, depth=4, num_classes=self.num_classes)

        else:
            raise Exception(f'Error. Backbone "{self.args.backbone}" not supported.')
        return backbone

    def forward(self, x):
        """
        Forward function of the model.
        """
        x = self.forward_feat(x)
        return x

    def training_step(self, batch, batch_idx, part='train'):
        x, y = batch
        logits = self(x)
        loss = cross_entropy(logits, y, smooth=self.args.label_smoothing)
        acc = accuracy(logits, y)

        # Log
        self.log(f'{part}_loss', loss)
        self.log(f'{part}_acc', acc, prog_bar=True)

        if part == 'train':
            self.log('lr', self.optimizer.param_groups[0]['lr'])
        """
        Define a single training step.
        """
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Define a single validation step.
        """
        return self.training_step(batch, batch_idx, part='val')

    def configure_optimizers(self):
        """
        Configure optimizer for the learning process.
        """
        if self.args.optimizer == 'adam':
            optimizer = Adam(self.parameters(), self.args.lr)
        elif self.args.optimizer == 'sgd':
            optimizer = SGD(self.parameters(), self.args.lr, momentum=0.9, weight_decay=5e-4)
        else:
            raise Exception(f'Error. Optimizer "{self.args.optimizer}" not supported.')
        self.optimizer = optimizer
        return optimizer


def get_model(args):
    """
    Return the Classifier model.
    """
    model_args = {
        'args': args,
    }
    model = Classifier(**model_args)
    if len(args.model_checkpoint) > 0:
        model = model.load_from_checkpoint(args.model_checkpoint, **model_args)
        print(f'Loaded model at "{args.model_checkpoint}"')
    return model
