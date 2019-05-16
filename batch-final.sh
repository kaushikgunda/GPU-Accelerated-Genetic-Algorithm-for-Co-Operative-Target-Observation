#!/bin/bash
#SBATCH -A research
#SBATCH -n 5
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END

module add cuda
nvcc honours-final.cu
c=12
d=24
h=150
for a in $(seq 5 5 25)
do
for b in $(seq 1 1 30) 
do
for e in $(seq 1 1 2)
do
for f in $(seq 1 1 2)
do
for g in $(seq 1 1 2)
do
./a.out $a $b $c $d $e $f $g $h >> output.txt
done
done
done
done
done
