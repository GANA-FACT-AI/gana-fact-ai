import argparse
import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from adversary.inversion import Inversion
from adversary.inference import Inference
from model.privacymodel import PrivacyModel
from adversary.angle_pred import AnglePred
from datasets import load_data


def train(args):
    os.makedirs(args.log_dir, exist_ok=True)

    train_loader, test_loader = load_data(args.dataset, args.batch_size, args.num_workers)

    # train_loader, test_loader = load_data(args.batch_size, args.num_workers)

    logger = TensorBoardLogger("logs", name=args.attack_model)

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
                         limit_val_batches=0.05,
                         val_check_interval=0.20
                         )

    trainer.logger._default_hp_metric = None

    pl.seed_everything(args.seed)  # To be reproducible
    privacy_model = PrivacyModel.load_from_checkpoint(args.checkpoint, hyperparams=args)

    # Inversion attacks
    if args.attack_model == 'inversion1':
        angle_pred_model = AnglePred.load_from_checkpoint(args.checkpoint_angle_pred, 
                                                                privacy_model=privacy_model)
        adversary_model = Inversion(privacy_model, angle_pred_model)
    elif args.attack_model == 'inversion2':
        adversary_model = Inversion(privacy_model)

    # Inference attacks
    elif args.attack_model == 'inference1':
        # angle_pred_model = AnglePred.load_from_checkpoint(args.checkpoint_angle_pred, 
        #                                                         privacy_model=privacy_model)
        # inversion_model = Inversion.load_from_checkpoint(args.checkpoint_inversion, 
        #                                                     privacy_model=privacy_model, 
        #                                                     angle_pred_model=angle_pred_model)
        adversary_model = Inference()
    elif args.attack_model == 'inference2':
        pass
    elif args.attack_model == 'inference3':
        pass
    elif args.attack_model == 'inference4':
        pass
    
    trainer.fit(adversary_model, train_loader, val_dataloaders=test_loader)

    # Testing
    #model = model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    #test_result = trainer.test(model, test_dataloaders=test_loader, verbose=True)


if __name__ == '__main__':
    # Feel free to add more argument parameters
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model hyperparameters
    parser.add_argument('--attack_model', default='inversion1', type=str,
                        help='What type of attack should be performed.')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        help='Dataset to train the model on.')

    # Optimizer hyperparameters
    parser.add_argument('--lr_model', default=1e-3, type=float)
    parser.add_argument('--lr_gen', default=1e-4, type=float)
    parser.add_argument('--lr_crit', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=40, type=int,
                        help='Max number of epochs.')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results.')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='Number of workers to use in the data loaders. To have a truly deterministic run, this has to be 0. ' + \
                             'For your assignment report, you can use multiple workers (e.g. 4) and do not have to set it to 0.')
    parser.add_argument('--log_dir', default='logs/adversary', type=str,
                        help='Directory where the PyTorch Lightning logs should be created.')
    parser.add_argument('--progress_bar', action='store_true',
                        help=('Use a progress bar indicator for interactive experimentation. '
                              'Not to be used in conjuction with SLURM jobs'))
    parser.add_argument('--debug', default=False, type=bool,
                        help='Shorten epochs and epoch lengths for quick debugging')
    parser.add_argument('--plot_graph', default=False, type=bool)
    parser.add_argument('--checkpoint', default='logs/lightning_logs/version_14/checkpoints/epoch=499.ckpt', type=str)
    parser.add_argument('--checkpoint_angle_pred', default='logs/angle_predictor/version_5/checkpoints/epoch=39.ckpt', type=str)
    parser.add_argument('--predict_angle', default=True, type=bool)

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
