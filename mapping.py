import os
from time import time
import argparse
import numpy as np

argparser = argparse.ArgumentParser()
argparser.add_argument('binsize', type=int)
argparser.add_argument('shape', choices=['sphere', 'circle', 'flower'])
argparser.add_argument('method', default='toeplitz', choices=['sinkhorn', 'toeplitz'])
argparser.add_argument('--inverse', action='count', required=False)
argparser.add_argument('--path', default='.', type=str, required=False)
args = argparser.parse_args()

# ------------------------------------

binsize, shape, method, inverse, path = args.binsize, args.shape, args.method, args.inverse, args.path

bins = np.load(os.path.join(path, f'bins_{shape}_{binsize}.npy'), allow_pickle=True)
p    = np.load(os.path.join(path, f'p_{shape}_{binsize}.npy'), allow_pickle=True)
q    = np.load(os.path.join(path, f'q_{shape}_{binsize}.npy'), allow_pickle=True)

K = np.load(os.path.join(path, f'K_{method}.npy'), allow_pickle=True).item(0)
a = np.load(os.path.join(path, f'a_{method}.npy'), allow_pickle=True)[:, None]
b = np.load(os.path.join(path, f'b_{method}.npy'), allow_pickle=True)[:, None]

start = time()
if method == 'toeplitz':
    if not inverse:
        bt = b * bins
        mapped = (np.vstack([K.matvec(bt[:,i], 0) for i in range(bins.shape[1])]) * a.T).T / (p.ravel() + 1e-20)[:,None]
    else:
        at = a * bins
        mapped = (np.vstack([K.matvec(at[:,i], 0) for i in range(bins.shape[1])]) * b.T).T / (q.ravel() + 1e-20)[:,None]
else:
    if not inverse:
        mapped = ((K   @ (b * bins[q.ravel()>0])).T * a.T).T / (p.ravel() + 1e-20)[p.ravel() > 0,None]
    else:
        mapped = ((K.T @ (a * bins[p.ravel()>0])).T * b.T).T / (q.ravel() + 1e-20)[q.ravel() > 0,None]
t = time() - start

name = os.path.join(path, f'{"source_image" if not inverse else "target_preimage"}_{method}_{shape}_{binsize}.npy')
np.save(name, mapped)
print(f'Done in {t*1000:.3f} ms.\nFile saved as {name}.')
