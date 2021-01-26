import argparse
import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from data.cub2011 import CUB
from data.CIFAR10 import load_data
from model.privacymodel import PrivacyModel
from pathlib import Path
import pandas as pd


def train(args):
    os.makedirs(args.log_dir, exist_ok=True)
    PATH = Path('./datasets/CUB_200_2011/CUB_200_2011/')

    labels = pd.read_csv(PATH/"image_class_labels.txt", header=None, sep=" ")
    labels.columns = ["id", "label"]

    train_test = pd.read_csv(PATH/"train_test_split.txt", header=None, sep=" ")
    train_test.columns = ["id", "is_train"]
    
    images = pd.read_csv(PATH/"images.txt", header=None, sep=" ")
    images.columns = ["id", "name"]

    classes = pd.read_csv(PATH/"classes.txt", header=None, sep=" ")
    classes.columns = ["id", "class"]

    train_dataset = CUB(PATH, labels, train_test, images, train=True, transform=True)
    test_dataset = CUB(PATH, labels, train_test, images, train=False, transform=False)

    #train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    #test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    train_loader, test_loader = load_data(args.batch_size, args.num_workers)

    # trainset = Cub2011(root='datasets/CUB-200/')
    # train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
    #                                         shuffle=True, num_workers=args.num_workers, drop_last=True)
    # testset = Cub2011(root='datasets/CUB-200/', train=False)
    # test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
    #                                         shuffle=True, num_workers=args.num_workers, drop_last=True)
    
    logger = TensorBoardLogger("logs", name="lightning_logs")

    trainer = pl.Trainer(default_root_dir=args.log_dir,
                         checkpoint_callback=args.checkpoint_callback,
                         gpus=1 if torch.cuda.is_available() else 0,
                         max_epochs=args.epochs,
                         logger=logger,
                         callbacks=[],
                         progress_bar_refresh_rate=1 if args.progress_bar else 0,
                         fast_dev_run=args.fast_dev_run,
                         overfit_batches=args.overfit_batches,
                         weights_summary=args.weights_summary,
                         limit_train_batches=args.limit_train_batches,
                         limit_val_batches=args.limit_val_batches,
                         #resume_from_checkpoint='./logs/lightning_logs/version_0/checkpoints/epoch=29.ckpt',
                         deterministic=True,
                         track_grad_norm=2
                         )
    trainer.logger._default_hp_metric = None

    pl.seed_everything(args.seed)  # To be reproducible
    model = PrivacyModel(args)

    trainer.fit(model, train_loader)

    # Testing
    #model = model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    #test_result = trainer.test(model, test_dataloaders=test_loader, verbose=True)


if __name__ == '__main__':
    # Feel free to add more argument parameters
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model hyperparameters
    parser.add_argument('--model', default='default', type=str,
                        help='What model to use in the VAE',
                        choices=['default'])

    # Optimizer hyperparameters
    parser.add_argument('--lr_gen', default=1e-4, type=float)
    parser.add_argument('--lr_crit', default=1e-4, type=float)
    parser.add_argument('--lr_model', default=1e-2, type=float)
    parser.add_argument('--batch_size', default=12, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=1, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='Number of workers to use in the data loaders. To have a truly deterministic run, this has to be 0. ' + \
                             'For your assignment report, you can use multiple workers (e.g. 4) and do not have to set it to 0.')
    parser.add_argument('--log_dir', default='logs', type=str,
                        help='Directory where the PyTorch Lightning logs should be created.')
    parser.add_argument('--progress_bar', action='store_true',
                        help=('Use a progress bar indicator for interactive experimentation. '
                              'Not to be used in conjuction with SLURM jobs'))
    parser.add_argument('--debug', default=False, type=bool,
                        help='Shorten epochs and epoch lengths for quick debugging')
    parser.add_argument('--plot_graph', default=False, type=bool)

    args = parser.parse_args()

    if args.debug:
        args.fast_dev_run = False
        args.overfit_batches = 10
        args.weights_summary = 'full'
        args.limit_train_batches = 20
        args.limit_val_batches = 20
        args.checkpoint_callback = False
    else:
        args.fast_dev_run = False
        args.overfit_batches = 0.0
        args.weights_summary = None
        args.limit_train_batches = 1.0
        args.limit_val_batches = 1.0
        args.checkpoint_callback = True

    train(args)
