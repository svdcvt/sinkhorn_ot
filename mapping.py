import os
from time import time
import argparse
import numpy as np

argparser = argparse.ArgumentParser(description=
                                    '''The program finds mapping gamma@Y with defined by arguments parameters. 
                                    Please, run the script from the directory with the following precomputed files:
                                    1) bins_{shape}_{binsize}.npy
                                    2) p_{shape}_{binsize}.npy
                                    3) q_{shape}_{binsize}.npy
                                    4) K_{method}.npy
                                    5) a_{method}.npy
                                    6) b_{method}.npy.
                                    
                                    These files can be precomputed with mesh_task.py (1-3) and sinkhorn.py (4-6).''')
argparser.add_argument('--binsize', type=int, help='number of bins in each dimension')
argparser.add_argument('--shape', choices=['ball', 'disk', 'flower'], help='task to solve')
argparser.add_argument('--method', default='toeplitz', choices=['sinkhorn', 'toeplitz'],  required=False, help='method to use (default toeplitz)')
argparser.add_argument('--inverse', action='count', required=False, help='whether to find mapping from target to source with the same plan')
argparser.add_argument('--path', default='.', type=str, required=False, help='dir to save obtained mapping (default ".")')
args = argparser.parse_args()

# ------------------------------------

binsize, shape, method, inverse, path = args.binsize, args.shape, args.method, args.inverse, args.path

try:
    bins = np.load(f'bins_{shape}_{binsize}.npy', allow_pickle=True)
    p    = np.load(f'p_{shape}_{binsize}.npy', allow_pickle=True)
    q    = np.load(f'q_{shape}_{binsize}.npy', allow_pickle=True)

    K = np.load(f'K_{method}.npy', allow_pickle=True).item(0)
    a = np.load(f'a_{method}.npy', allow_pickle=True)[:, None]
    b = np.load(f'b_{method}.npy', allow_pickle=True)[:, None]
except:
    print('Files with arrays for algorithm are not found (bins, p, q; K, a, b),'\
          'please, run script from the directory with these files.')
    exit()

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
