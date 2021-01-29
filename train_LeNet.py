import argparse
import os

import torch
import pytorch_lightning as pl

# from data.CIFAR10 import load_data
from datasets import load_data
from LeNet.privacymodel import PrivacyModel


def train(args):
    os.makedirs(args.log_dir, exist_ok=True)

    train_loader, test_loader = load_data(args.dataset, args.batch_size, args.num_workers, adversary=False)

    trainer = pl.Trainer(default_root_dir=args.log_dir,
                         checkpoint_callback=True,
                         gpus=1 if torch.cuda.is_available() else 0,
                         max_epochs=args.epochs,
                         callbacks=[],
                         progress_bar_refresh_rate=1 if args.progress_bar else 0,
                         fast_dev_run=args.fast_dev_run,
                         overfit_batches=args.overfit_batches,
                         weights_summary=args.weights_summary,
                         limit_train_batches=args.limit_train_batches,
                         limit_val_batches=args.limit_val_batches,
                         # resume_from_checkpoint='./logs/lightning_logs/version_0/checkpoints/epoch=29.ckpt',
                         deterministic=True
                         )

    pl.utilities.seed.seed_everything(args.seed)  # To be reproducible

    #
    model = PrivacyModel()
    trainer.fit(model, train_loader)
    #
    # Testing
    # model = PrivacyModel().load_from_checkpoint(args.checkpoint, hyperparams=args, strict=False)
    # trainer.test(model, test_dataloaders=test_loader, verbose=True)

    # print(trainer.test(model, test_dataloaders=test_loader, verbose=True))
    # trainer.test(model, test_dataloaders=test_loader, verbose=True)

    # model = model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    # test_result = trainer.test(model, test_dataloaders=test_loader, verbose=True)


if __name__ == '__main__':
    # Feel free to add more argument parameters
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model hyperparameters
    parser.add_argument('--model', default='default', type=str,
                        help='What model to use in the VAE',
                        choices=['default'])
    parser.add_argument('--dataset', default='cifar10', type=str,
                        help='Dataset to train the model on.')

    # Optimizer hyperparameters
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=500, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers to use in the data loaders. To have a truly deterministic run, this has to be 0. ' + \
                             'For your assignment report, you can use multiple workers (e.g. 4) and do not have to set it to 0.')
    parser.add_argument('--log_dir', default='logs', type=str,
                        help='Directory where the PyTorch Lightning logs should be created.')
    parser.add_argument('--progress_bar', action='store_true',
                        help=('Use a progress bar indicator for interactive experimentation. '
                              'Not to be used in conjuction with SLURM jobs'))
    parser.add_argument('--debug', default=False, type=float,
                        help='Shorten epochs and epoch lengths for quick debugging')
    parser.add_argument('--checkpoint',
                        default='logs/lightning_logs/privacy_cifar10/checkpoints/epoch=499-step=97499.ckpt', type=str)
    args = parser.parse_args()

    if args.debug:
        args.fast_dev_run = 5
        args.overfit_batches = 10
        args.weights_summary = 'full'
        args.limit_train_batches = 20
        args.limit_val_batches = 20
    else:
        args.fast_dev_run = False
        args.overfit_batches = 0.0
        args.weights_summary = None
        args.limit_train_batches = 1.0
        args.limit_val_batches = 1.0

    train(args)
