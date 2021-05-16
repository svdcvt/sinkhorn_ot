import os
from time import time
import argparse
import numpy as np

from sinkhorn_utils import sinkhorn, sinkhorn_toeplitz

argparser = argparse.ArgumentParser()
argparser.add_argument('beta', type=float)
argparser.add_argument('binsize', type=int)
argparser.add_argument('shape', choices=['sphere', 'circle', 'flower'])
argparser.add_argument('method', default='toeplitz', choices=['sinkhorn', 'toeplitz'])
argparser.add_argument('--num_iter', type=int, default=100, required=False)
argparser.add_argument('--path', default='.', type=str, required=False)
args = argparser.parse_args()

try:
    bins = np.load(f'bins_{args.shape}_{args.binsize}.npy', allow_pickle=True)
    p = np.load(f'p_{args.shape}_{args.binsize}.npy', allow_pickle=True)
    q = np.load(f'q_{args.shape}_{args.binsize}.npy', allow_pickle=True)
except:
    print("LOL")

if method == 'toeplitz':
    K, a, b, _, _, _ = sinkhorn_toeplitz(p, q, bin_size=bins, beta=args.beta, max_iter=args.num_iter, 
                                                  eps=1e-20, tol=1e-20)
elif method == 'sinkhorn':
    X, Y = bins[p.ravel() > 0], bins[q.ravel() > 0]
    K, a, b = sinkhorn(X, Y, beta=beta, max_iter=args.num_iter, eps=1e-20, tol=1e-20)

np.save(os.path.join(path, f'K_{method}.npy'), K)
np.save(os.path.join(path, f'a_{method}.npy'), a)
np.save(os.path.join(path, f'b_{method}.npy'), b)
