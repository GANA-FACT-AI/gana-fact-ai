#!/usr/bin/env bash

rsync -av . lgpu0160@lisa.surfsara.nl:~/code/ --exclude=datasets --exclude=logs --exclude=.git --exclude=.idea

ssh lgpu0160@lisa.surfsara.nl 'sbatch $HOME/code/slurm.sh'
