import os
from time import time
import argparse
import numpy as np

from sinkhorn_utils import sinkhorn, sinkhorn_toeplitz

argparser = argparse.ArgumentParser(description=
                                    '''The program runs sinkhorn algorithm with defined by arguments parameters. 
                                    Please, run the script from the directory with the following precomputed files:
                                    1) bins_{shape}_{binsize}.npy
                                    2) p_{shape}_{binsize}.npy
                                    3) q_{shape}_{binsize}.npy.
                                    
                                    These files can be precomputed with mesh_task.py.''')
argparser.add_argument('beta', type=float, help='regularization parameter')
argparser.add_argument('binsize', type=int, help='number of bins in each dimension')
argparser.add_argument('shape', choices=['ball', 'disk', 'flower'], help='task to solve')
argparser.add_argument('method', default='toeplitz', choices=['sinkhorn', 'toeplitz'], help='method to use (default toeplitz)')
argparser.add_argument('--num_iter', default=100, type=int, required=False, help='maximum number of iterations (default 100)')
argparser.add_argument('--path', default='.', type=str, required=False, help='path to save obtained matrices (default ".")')
args = argparser.parse_args()

try:
    bins = np.load(f'bins_{args.shape}_{args.binsize}.npy', allow_pickle=True)
    p = np.load(f'p_{args.shape}_{args.binsize}.npy', allow_pickle=True)
    q = np.load(f'q_{args.shape}_{args.binsize}.npy', allow_pickle=True)
except:
    print('Files with arrays for algorithm are not found (bins, p, q), please, run script from the directory with these files.')
    exit()

if args.method == 'toeplitz':
    K, a, b, _, _, _ = sinkhorn_toeplitz(p, q, bin_size=bins, beta=args.beta, max_iter=args.num_iter, 
                                                  eps=1e-20, tol=1e-20, plot=1)
elif args.method == 'sinkhorn':
    X, Y = bins[p.ravel() > 0], bins[q.ravel() > 0]
    K, a, b = sinkhorn(X, Y, beta=args.beta, max_iter=args.num_iter, eps=1e-20, tol=1e-20)

np.save(os.path.join(args.path, f'K_{args.method}.npy'), K)
np.save(os.path.join(args.path, f'a_{args.method}.npy'), a)
np.save(os.path.join(args.path, f'b_{args.method}.npy'), b)
