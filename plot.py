import plot_utils
import argparse
import numpy as np
import os

argparser = argparse.ArgumentParser(description=
                                    '''The program runs sinkhorn algorithm with defined by arguments parameters. 
                                    Please, run the script from the directory with the following precomputed files:
                                    1) 
                                    2) bins_{shape}_{binsize}.npy
                                    3) p_{shape}_{binsize}.npy
                                    4) q_{shape}_{binsize}.npy.
                                    
                                    These files can be precomputed with mesh_task.py.''')
argparser.add_argument('beta', type=float, help='used regularization parameter (for titles)')
argparser.add_argument('binsize', type=int, help='used number of bins')
argparser.add_argument('shape', choices=['ball', 'disk', 'flower'], help='solved task')
argparser.add_argument('--method', default='toeplitz', choices=['sinkhorn', 'toeplitz'], required=False, help='used method (default toeplitz)')
argparser.add_argument('--limits', nargs="*", required=False, help='limits for plots, pass 4 or 6 arguments for min_x, min_y, [min_z], max_x, max_y, [max_z] (default target min/max)')
argparser.add_argument('--inverse', action='count', required=False, help='whether to plot mapping from target to source with the same plan (default not inverse)')
argparser.add_argument('--path', default='../images/', type=str, required=False, help='dir to save images (default "./images/")')
args = argparser.parse_args()
if args.limits is not None:
    if len(args.limits) in [4, 6]:
        args.limits = tuple((float(args.limits[i]), float(args.limits[i+1])) for i in range(0, len(args.limits), 2))
    else:
        raise argparse.ArgumentError(f'limits: pass 4 or 6 arguments for min_x, min_y, [min_z], max_x, max_y, [max_z] (passed {len(arg.limits)})')


try:
    image = np.load(f'{"source_image" if not args.inverse else "target_preimage"}_{args.method}_{args.shape}_{args.binsize}.npy', allow_pickle=True)
    distribution = np.load(f'p_{args.shape}_{args.binsize}.npy', allow_pickle=True) if not args.inverse else np.load(f'q_{args.shape}_{args.binsize}.npy', allow_pickle=True)
    bins = np.load(f'bins_{args.shape}_{args.binsize}.npy', allow_pickle=True)
    target_d = np.load(f'q_{args.shape}_{args.binsize}.npy', allow_pickle=True) if not args.inverse else np.load(f'p_{args.shape}_{args.binsize}.npy', allow_pickle=True)
    target = bins[target_d.ravel() > 0]
except:
    print(f'Files with arrays for algorithm are not found ({"source_image" if not args.inverse else "target_preimage"}, bins, p, q), '\
           'please, run script from the directory with these files.')
    exit()

# print(image)
# exit()
plot_utils.plot_image(target, image, path=os.path.join(args.path, 'image'),
                      method=args.method, shape=args.shape, binsize=args.binsize, 
                      beta=args.beta, limits=args.limits)
if args.shape in ['disk', 'flower']:
    plot_utils.plot_mesh(image, distribution, path=os.path.join(args.path, 'mesh'), every=1, 
                         method=args.method, shape=args.shape, binsize=args.binsize, 
                         beta=args.beta, limits=args.limits)
elif args.shape in ['ball', 'tor']:
    plot_utils.plot_3d_mesh(image, distribution, path=os.path.join(args.path, 'mesh'), every=1, animate=False,
                            method=args.method, shape=args.shape, binsize=args.binsize, 
                            beta=args.beta, limits=args.limits)