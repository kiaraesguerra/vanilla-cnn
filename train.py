import argparse
import torch
from dataloader import get_dataloaders
from utils import get_model
from pruning_utils import *
from train_pl import return_pl_module
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import models


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10')

parser.add_argument('--model', '-a', default='van32')
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--eval-batch-size', type=int, default=100, metavar='N')
parser.add_argument('--epochs', default=200, type=int, metavar='N')
parser.add_argument('--seed', type=int, default=3407)
parser.add_argument('--pretrained', dest='pretrained', action='store_true')

parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--nesterov', default=True, type=bool)
parser.add_argument('--milestones', default=[100, 150], type=list)
parser.add_argument('--weight-decay', default=1e-4, type=float)
parser.add_argument('--gamma', default=0.1, type=float)
parser.add_argument('--optimizer', type=str, default='sgd')
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.99)
parser.add_argument('--warmup-epoch', type=int, default=5)
parser.add_argument('--scheduler', type=str, default='step')


parser.add_argument('--ckpt_path', type=str, default=None)
parser.add_argument('--ckpt_path_resume', type=str, default=None)
parser.add_argument('--conv-init', type=str, default='conv_delta_orthogonal')
parser.add_argument('--experiment-name', type=str, default=None)
parser.add_argument('--autoaugment', type=bool, default=False)
parser.add_argument('--cutmix', type=bool, default=False,)
parser.add_argument('--width', type=int, default=16)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--label-smoothing', type=float, default=0)

parser.add_argument('--pruning-method', type=str, default=None, choices=['LMP-S', 'SAO', 'SAO-relu',
                                                                            'RG-S'])
parser.add_argument('--degree', type=int, default=None)
parser.add_argument('--sparsity', type=float, default=None)
parser.add_argument('--activation', type=str, default='tanh')
parser.add_argument('--dirpath', type=str, default=None)

args = parser.parse_args()


if __name__=='__main__':
    train_dl, test_dl = get_dataloaders(args)
    model = get_model(args)
    
    
    if args.pruning_method:
        print(args.pruning_method)
        if args.degree:
            print(args.degree)
            model = custom_pruner_degree(model, args.pruning_method, args.degree)
        elif args.sparsity:
            print(args.sparsity)
            model = custom_pruner(model, args.pruning_method, args.sparsity)
            
        
    model = return_pl_module(model, args)
    checkpoint_callback = ModelCheckpoint(monitor='val/acc', save_top_k=1, 
                        auto_insert_metric_name=False, save_last=True, filename='best', save_on_train_epoch_end=True, 
                        dirpath=f'{args.dirpath}/Results/{args.experiment_name}', verbose=True)
    logger = CSVLogger(f"{args.dirpath}/Results/{args.experiment_name}", name="logs")
    trainer = Trainer(max_epochs=args.epochs, accelerator='gpu', callbacks=[checkpoint_callback], logger=logger, resume_from_checkpoint=args.ckpt_path_resume)
    trainer.fit(model, train_dl, test_dl, ckpt_path=args.ckpt_path)
    
    ckpt_path = checkpoint_callback.best_model_path
    model_checkpoint = torch.load(ckpt_path)
    model.load_state_dict(model_checkpoint["state_dict"])
    if args.pruning_method:
        remove_parameters(model)
    torch.save(model.state_dict(), f'{args.dirpath}/Results/{args.experiment_name}/{args.experiment_name}.pt')