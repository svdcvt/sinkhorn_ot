import numpy as np
import os
from sinkhorn_utils import binning, sinkhorn, sinkhorn_toeplitz

binsizes = [np.logspace(3, 12, 10, dtype=int, base=2),
            np.logspace(3, 8, 6, dtype=int, base=2)]
print(binsizes)

tasks = [(2, 'disk'), (3, 'ball')]

side = 1
radius = 1
beta = 0.05
eps = 1e-8
niter = 20
path = './arrays/'

# # 1) generate tasks arrays

for d, shape in tasks:
    for binsize in binsizes[d-2]:
        borders = np.array(np.meshgrid(*([-side, side],)*d)).reshape(-1, d)
        bins, p, q = binning(borders, borders, binsize)
        x, y = bins[:, 0], bins[:, 1]

        # p
        p = np.zeros_like(p).ravel()
        is_reg = np.all(np.abs(bins) <= side, axis=1)
        p[is_reg] = 1 / is_reg.sum()
        p = p.reshape(*(binsize,)*d)
        # q
        q = np.zeros_like(q).ravel()
        is_shape = np.sum(bins ** 2, axis=1) <= radius ** 2
        q[is_shape] = 1 / is_shape.sum()
        q = q.reshape(*(binsize,)*d)

        np.save(os.path.join(path, f'bins_{shape}_{binsize}.npy'), bins)
        np.save(os.path.join(path, f'p_{shape}_{binsize}.npy'), p)
        np.save(os.path.join(path, f'q_{shape}_{binsize}.npy'), q)

T_all_ts = [[[] for j in range(len(binsizes[i]))] for i in range(len(tasks))]
T_all_s =  [[[] for j in range(len(binsizes[i]))] for i in range(len(tasks))]

# 2) test S and TS

for j, (d, shape) in enumerate(tasks):
    print('----- TASK =', shape)
    for i, binsize in enumerate(binsizes[d-2]):
        print('----- BINS =', binsize)
        bins = np.load(os.path.join(path, f'bins_{shape}_{binsize}.npy'), allow_pickle=True)
        p = np.load(os.path.join(path, f'p_{shape}_{binsize}.npy'), allow_pickle=True)
        q = np.load(os.path.join(path, f'q_{shape}_{binsize}.npy'), allow_pickle=True)

        for _ in range(100 if binsize):
            _, _, _, _, _, _, T = sinkhorn_toeplitz(p, q, bin_size=bins, beta=beta, 
                                                    max_iter=niter, plot=0, store_err=0, testtime=True)
            T_all_ts[j][i].append(T)

flag = 0
for j, (d, shape) in enumerate(tasks):
    print('----- TASK =', shape)
    for i, binsize in enumerate(binsizes[d-2]):
        print('----- BINS =', binsize)
        bins = np.load(os.path.join(path, f'bins_{shape}_{binsize}.npy'), allow_pickle=True)
        p = np.load(os.path.join(path, f'p_{shape}_{binsize}.npy'), allow_pickle=True)
        q = np.load(os.path.join(path, f'q_{shape}_{binsize}.npy'), allow_pickle=True)
        X, Y = bins[p.ravel() > 0], bins[q.ravel() > 0]

        for _ in range(50):
            try:
                _, _, _, T = sinkhorn(X, Y, beta=beta, max_iter=niter, plot_err=0, store_err=0, testtime=True)
                niter = len(T)
                T_all_s[j][i].append(T)
            except Exception as e:
                print(e)
                flag = 1
                break
        if flag:
            break

np.save('results.npy', [T_all_ts, T_all_s])
# np.load('results.npy', allow_pickle=True).item(0)