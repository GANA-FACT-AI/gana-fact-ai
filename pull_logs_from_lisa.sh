#!/usr/bin/env bash

rsync -av lgpu0160@lisa.surfsara.nl:~/code/logs .

tensorboard --logdir logs/lightning_logs