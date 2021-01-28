#!/usr/bin/env bash

rsync -av lgpu0160@lisa.surfsara.nl:~/code/logs/adversary/version_24/ logs/adversary/swap_mae/

tensorboard --logdir logs/lightning_logs