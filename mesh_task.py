import os
from time import time
import argparse
import numpy as np

from sinkhorn_utils import binning
from plot_utils import plot_target

argparser = argparse.ArgumentParser(description=
                                    '''The program creates arrays for source and target shape: bins, p, q. 
                                    The target shape defines by arguments. The distributions are uniform on grid, 
                                    defined on interval [-side, side]^{2|3} with number of bins defined by binsize parameter.
                                     
                                    Since files can be large, arrays are saved with names defined only by shape and binsize.''')
argparser.add_argument('--binsize', type=int, help='number of bins in each dimension')
argparser.add_argument('--shape', choices=['ball', 'disk', 'flower'], help='task to solve')
argparser.add_argument('-ABT', default=[0.7, 0.3, 3], nargs=3, type=float, required=False, help='flower parameters: A + B * cos(T*theta) (default 0.7 0.3 3)')
argparser.add_argument('-r', '--radius', default=1., type=float, required=False, help='ball/disk radius (default 1.)')
argparser.add_argument('-s', '--side', default=1., type=float, required=False, help='square/cube size: [-s, s]^{2|3} (default 1.)')
argparser.add_argument('-o', '--outer', default=0, action='count', required=False, help='whether to use outer of target shape (крыло)')
argparser.add_argument('--path', default='./arrays', type=str, required=False, help='dir path to save (default ./arrays)')
args = argparser.parse_args()

os.makedirs(args.path, exist_ok=True)
binsize, shape = args.binsize, args.shape
A, B, t = args.ABT
while A + B > args.side and shape == 'flower':
    print('For flower (target) shape it is recommended to use A and B parameters'\
            'such that A + B <= side, since square (source) shape is [-side, side]^2.')
    A, B, args.side = map(float, input('Input another values for A B S (half-side):').split())
while args.radius > args.side and shape in ['disk', 'ball']:
    print('For disk (target) shape it is recommended to use radius less or equal than square (target) half-side.')
    args.radius, args.side = map(float, input('Input another values for R (radius) and S (half-side):').split())

if shape in ['disk', 'flower']:
    # 2D shape
    start = time()
    borders = np.array(np.meshgrid([-args.side, args.side],
                                   [-args.side, args.side])).reshape(-1, 2)
    bins, p, q = binning(borders, borders, binsize)
    x, y = bins[:, 0], bins[:, 1]

    if shape == 'disk': 
        is_shape = x ** 2 + y ** 2  <= args.radius ** 2
    elif shape == 'flower':
        is_shape = (A + B * np.cos(t * np.arctan2(y, x))) >= np.sqrt(x ** 2 + y ** 2)
    
    # p
    p = np.zeros_like(p).ravel()
    is_reg = (np.abs(x) <= args.side) * (np.abs(y) <= args.side)
    p[is_reg] = 1 / is_reg.sum()
    p = p.reshape(binsize, binsize)
    # q
    q = np.zeros_like(q).ravel()
    q[is_shape if not args.outer else ~is_shape] = 1 / is_shape.sum()
    q = q.reshape(binsize, binsize)

    end = time() - start

elif shape in ['ball', 'tor']:
    # 3D shape
    start = time()
    borders = np.array(np.meshgrid([-args.side, args.side],
                                   [-args.side, args.side],
                                   [-args.side, args.side])).reshape(-1, 3)
    bins, p, q = binning(borders, borders, binsize)
    x, y, z = bins[:, 0], bins[:, 1], bins[:, 2]

    if shape == 'ball':
        is_shape = x**2 + y**2 + z**2 <= args.radius**2
    elif shape == 'tor': # not a choice right now, since it has a hole and can't be meshed properly idky
        R, r = 0.7, 0.3
        is_shape = (x**2 + y**2 + z**2 + R**2 - r**2)**2 - 4 * R**2*(x**2 + y**2) <= 0
    # p
    p = np.zeros_like(p).ravel()
    is_reg = (np.abs(x) <= args.side) * (np.abs(y) <= args.side) * (np.abs(z) <= args.side)
    p[is_reg] = 1 / is_reg.sum()
    p = p.reshape(binsize, binsize, binsize)
    # q
    q = np.zeros_like(q).ravel()
    q[is_shape if not args.outer else ~is_shape] = 1 / is_shape.sum()
    q = q.reshape(binsize, binsize, binsize)

    end = time() - start

np.save(os.path.join(args.path, f'bins_{shape}_{binsize}.npy'), bins)
np.save(os.path.join(args.path, f'p_{shape}_{binsize}.npy'), p)
np.save(os.path.join(args.path, f'q_{shape}_{binsize}.npy'), q)

plot_target(q, args.side)
print(f'Done in {end*1000:.5f}ms.\nFiles saved at {args.path}. See target plot at "shape.png"')