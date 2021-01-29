#!/usr/bin/env bash

rsync -av lgpu0160@lisa.surfsara.nl:~/code/logs/lightning_logs/version_140/ logs/lightning_logs/version_140/

tensorboard --logdir logs/lightning_logs