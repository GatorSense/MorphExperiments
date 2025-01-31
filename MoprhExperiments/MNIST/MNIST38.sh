#!/bin/sh
#SBATCH --job-name=MNIST38   # Job name
#SBATCH --mail-type=ALL               # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=<weihuang.xu@ufl.edu>   # Where to send mail	
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --mem=4000mb                   # Memory limit
#SBATCH --time=100:00:00               # Time limit hrs:min:sec
#SBATCH --output=MNIST38.out   # Standard output and error log

pwd; hostname; date


echo "Running plot script on a single CPU core"

python /ufrc/azare/weihuang.xu/MNIST/MNIST_two_layer38.py

date
