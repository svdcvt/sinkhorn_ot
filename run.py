import os
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--beta', type=float)
argparser.add_argument('--eps', type=float)
argparser.add_argument('--binsize', type=int)
argparser.add_argument('-N', type=int)
argparser.add_argument('--shape')
args = argparser.parse_args()

meshsize = 8
N = args.N
beta = args.beta
shape = args.shape
binsize = args.binsize
eps = args.eps if args.eps is not None else 1e-12

os.system(f'python3 ../sinkhorn.py --beta {beta} --shape {shape} -N {N} --binsize {binsize} --eps {eps} -i')
os.system(f'python3 ../mapping.py --shape {shape} --binsize {binsize}')
os.system(f'python3 ../plot.py --beta {beta} --shape {shape} --binsize {binsize} --meshsize {meshsize}')