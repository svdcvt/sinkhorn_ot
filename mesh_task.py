import os
from time import time
import argparse
import numpy as np

from sinkhorn_utils import binning

argparser = argparse.ArgumentParser()
argparser.add_argument('binsize', type=int)
argparser.add_argument('d', choices=[2, 3])
argparser.add_argument('shape', choices=['sphere', 'circle', 'flower'])
argparser.add_argument('--path', default='.', type=str, required=False)
args = argparser.parse_args()

binsize, shape = args.binsize, args.shape

if d == 2:
    # 2D shape

    start = time()

    borders = np.array(np.meshgrid([-1, 1], [-1, 1])).reshape(-1, 2)
    bins, p, q = binning(borders, borders, binsize)
    x, y = bins[:, 0], bins[:, 1]
    A, B = 0.7, 0.3


    if shape == 'circle': 
        is_shape = x ** 2 + y ** 2  <= 1
    elif shape == 'flower':
        is_shape = (A + B * np.cos(3 * np.arctan2(y, x))) >= np.sqrt(x ** 2 + y ** 2) 
    finally:
        # p
        p = np.zeros_like(p).ravel()
        is_reg = (np.abs(x) <= 1) * (np.abs(y) <= 1)
        p[is_reg] = 1 / is_reg.sum()
        p = p.reshape(binsize, binsize)
        # q
        q = np.zeros_like(q).ravel()
        q[is_shape] = 1 / is_shape.sum()
        q = q.reshape(binsize, binsize)

    end = time() - start()

elif d == 3:
    # 3D shape

    start = time()

    borders = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).reshape(-1, 3)
    bins, p, q = binning(borders, borders, binsize)
    x, y, z = bins[:, 0], bins[:, 1], bins[:, 2]
    R, r = 0.7, 0.3

    if shape == 'sphere':
        is_shape = x**2 + y**2 + z**2 <= 1
    elif shape == 'tor':
        is_shape = (x**2 + y**2 + z**2 + R**2 - r**2)**2 - 4 * R**2*(x**2 + y**2) <= 0
    finally:
        # p
        p = np.zeros_like(p).ravel()
        is_reg = (np.abs(x) <= 1) * (np.abs(y) <= 1) * (np.abs(z) <= 1)
        p[is_reg] = 1 / is_reg.sum()
        p = p.reshape(binsize, binsize, binsize)
        # q
        q = np.zeros_like(q).ravel()
        q[is_shape] = 1 / is_shape.sum()
        q = q.reshape(binsize, binsize, binsize)

    end = time() - start()
else:
    raise argparse.ArgumentError()

np.save(os.path.join(args.path, f'bins_{shape}_{binsize}.npy'), bins)
np.save(os.path.join(args.path, f'p_{shape}_{binsize}.npy'), p)
np.save(os.path.join(args.path, f'q_{shape}_{binsize}.npy'), q)

print(f'Done in {end*1000}ms.\nFiles saved at {args.path}.')