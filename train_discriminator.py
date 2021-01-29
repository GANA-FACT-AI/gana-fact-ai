import argparse
import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from adversary.angle_pred import AnglePred
from datasets import load_data
from model.privacymodel import PrivacyModel
from resnet.resnet_privacy_model import ResNetPrivacyModel


def train(args):
    os.makedirs(args.log_dir, exist_ok=True)
    train_loader, test_loader = load_data(args.dataset, args.batch_size, args.num_workers)

    logger = TensorBoardLogger("logs", name="angle_predictor")

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
                         limit_val_batches=0.01,
                         val_check_interval=0.20
                         )
    trainer.logger._default_hp_metric = None

    pl.seed_everything(args.seed)  # To be reproducible
    privacymodel = ResNetPrivacyModel.load_from_checkpoint(args.checkpoint, hyperparams=args)
    model = AnglePred(privacymodel)

    #trainer.fit(model, train_loader, val_dataloaders=test_loader)

    # Testing
    model = AnglePred.load_from_checkpoint(args.checkpoint_angle_pred, privacy_model=privacymodel, strict=False)
    test_result = trainer.test(model, test_dataloaders=test_loader, verbose=True)


if __name__ == '__main__':
    # Feel free to add more argument parameters
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model hyperparameters
    parser.add_argument('--model', default='resnet110a', type=str,
                        help='Choose the model.')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        help='Dataset to train the model on.')
    parser.add_argument('--checkpoint',
                        default='logs/lightning_logs/version_70/checkpoints/resnet-110-alpha-cifar10.ckpt', type=str)
    parser.add_argument('--add_gen_conv', default=False, type=bool)

    # Optimizer hyperparameters
    parser.add_argument('--lr_gen', default=1e-4, type=float)
    parser.add_argument('--lr_crit', default=1e-4, type=float)
    parser.add_argument('--lr_model', default=1e-3, type=float)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')
    parser.add_argument('--beta1', default=0.5, type=float)
    parser.add_argument('--beta2', default=0.999, type=float)

    # Other hyperparameters
    parser.add_argument('--epochs', default=500, type=int,
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
    parser.add_argument('--lambda_', default=10, type=int)
    parser.add_argument('--random_swap', default=False, type=bool)
    parser.add_argument('--checkpoint_angle_pred',
                        default='logs/angle_predictor/version_6/checkpoints/angle-resnet-110-alpha-cifar10.ckpt',
                        type=str)

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
