from torch import nn, optim
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import torch
import warmup_scheduler
from torchmetrics import Accuracy


def return_pl_module(model, args):
    model = Model(model, args)
    return model


class Model(LightningModule):
    def __init__(self, model, args):
        super().__init__()
        
        self.model = model
        self.lr = args.lr
        self.epochs = args.epochs
        self.momentum = args.momentum
        self.weight_decay=args.weight_decay
        self.nesterov=args.nesterov
        self.beta1=0
        self.beta2=0.99
        self.criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        self.num_steps = 0
        self.gamma=args.gamma
        self.epoch_loss, self.epoch_corr, self.epoch_acc = 0., 0., 0.
        self.milestones = args.milestones
        
        
        if args.optimizer=='sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, 
                                       weight_decay=self.weight_decay, nesterov=self.nesterov)
        elif args.optimizer=='adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, betas=(self.beta1, self.beta2), 
                                        weight_decay=self.weight_decay)

        if args.scheduler=='step':
            self.base_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.milestones, 
                                                                 gamma=self.gamma)
        elif args.scheduler=='cosine':
            self.base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 
                                                                             T_max=self.epochs, eta_min=self.min_lr)
            
            
        if args.warmup_epoch:
            self.scheduler = warmup_scheduler.GradualWarmupScheduler(self.optimizer, multiplier=1., total_epoch=self.warmup_epoch, after_scheduler=self.base_scheduler)
        else:
            self.scheduler = self.base_scheduler

        self.val_accuracy = Accuracy(task="multiclass", num_classes=10)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=10)
    
    def forward(self, x):
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        loss = self.criterion(out, y)
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)

        self.log("val/loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/acc", self.val_accuracy, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return [self.optimizer], [{"scheduler": self.scheduler, "interval": "epoch"}]
    
    
    
    