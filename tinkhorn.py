import mkl_fft
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.spatial.distance import cdist
from tqdm import tqdm


def sinkhorn(X, Y, beta=0.01, max_iter=200, store_err=True, early_stopping=True, 
             eps=1e-12, tol=1e-10, patience=10, verbose=True, plot=True):
    '''
    X, Y - datapoint of two distributions
    beta - regularization parameter 
    '''
    n, m = len(X), len(Y)
    C = cdist(X, Y)
    
    p = np.ones(n) / n
    q = np.ones(m) / m
    
    K = np.exp(-C / beta)
    Kt = K.T
    b = np.ones(m)
    i = 0
    err = [10]
    j = 0
    for i in range(max_iter):
        Kb = K.dot(b)
        a = np.divide(p, Kb)
        Kta = Kt.dot(a)
        b = np.divide(q, Kta)
        if store_err:
#             g = np.diag(a) @ K @ np.diag(b)
#             err.append(np.linalg.norm(g.sum(0) - q) + np.linalg.norm(g.sum(1) - p))
            g0 = a * K.dot(b)
            g1 = b * Kt.dot(a)
            err.append(np.linalg.norm(g0 - p) + np.linalg.norm(g1 - q))
            if early_stopping:
                # if good enough
                if err[-1] < eps:
                    if verbose:
                        print(f'#iterations={i+1}, early stopping: eps, err={err[-1]}')
                    break
                # if no improvements
                if np.abs(err[-2] - err[-1]) < tol:
                    j += 1
                    if j > patience:
                        if verbose:
                            print(f'#iterations={i+1}, early stopping: tol, err={err[-1]}')
                        break
                else:
                    j = 0
    if plot:
        plt.figure(figsize=(10,4))
        if store_err:
            plt.subplot(121)
            plt.title('error')
            plt.semilogy(range(len(err)-1), err[1:])
            plt.subplot(122)
        plt.title('optimal transport matrix')
        plt.imshow(a.reshape(-1,1) * K * b.reshape(1,-1))
        plt.show()
    
    return K, a, b

centers = lambda edges: (edges[:,:-1] + edges[:,1:]) / 2

def binning(X, Y, bin_size):
    '''
    X, Y: nd.array,
            data points of two distributions
    bin_size: int,
            number of bins (equal for each dimension!)
    '''
    
    clouds = np.vstack([X, Y])
    
    grid = np.linspace(np.min(clouds, 0), np.max(clouds, 0), bin_size + 1).T # [D, B] n+m +b
    
    mesh = np.meshgrid(*centers(grid), indexing='xy') 
    bins = np.hstack([x.reshape(-1,1) for x in mesh])
    
    p, _ = np.histogramdd(X, bins=grid)
    q, _ = np.histogramdd(Y, bins=grid)
    p /= p.sum()
    q /= q.sum()
    
    return bins, p, q

is_pow_2 = lambda x: x == 2 ** int(np.log2(x))

class Toeplitz(object):
    '''Class for utilizing Toeplitz matrix (assume it is symmetric and has block-level no more than 3)'''
    def __init__(self, top, size=None):
        '''
        
        top: nd.array, 
                the first row of the matrix, that defines the matrix; if 'size' is None, then assume that 
                shape of the array defines the block-level structure.
        size: tuple of int or None, 
                defines block-level structure, i.e. number of dimensions equals to level of matrix,
                each size of dimension is number of blocks of each level and size of last dimension is block size.
                Any size needs to be a power of 2.
        '''
        self.top = top if size is None else top.reshape(size)
        self.size = top.shape if size is None else size
        self.dim = len(self.size)
        self.area = tuple(slice(s) for s in self.size)
        self.pad = tuple((0, s) for s in self.size)
        
        assert all([is_pow_2(s) for s in self.size]), 'sizes need to be powers of 2'
        
        self.embedding = self.make_embedding(self.top)
        self.circ_fft = mkl_fft.fftn(self.embedding)
    
    def make_embedding(self, top):
        blocks = top
        for i in range(-1, -self.dim - 1, -1):
            zeros_size = list(self.size)
            for j in range(i, 0):
                if j == i:
                    zeros_size[j] = 1
                else:
                    zeros_size[j] *= 2
            slice_ = [slice(None) for _ in range(self.dim)]
            slice_[i] = slice(None, 0, -1)
            blocks = np.concatenate([blocks, np.zeros(zeros_size), blocks[tuple(slice_)]], i)
        return blocks
        
    def matvec(self, x):
        ''' fast "matvec" multiplication '''
        if x.ndim > 1:
            if (x.shape[0] == 1 or x.shape[1] == 1):
                x = x.ravel()
            else:
                raise ValueError()
        x_fft = mkl_fft.fftn(np.pad(x.reshape(self.size), self.pad))
        
        return np.abs(mkl_fft.ifftn(self.circ_fft * x_fft)[self.area]).ravel()
    
    def full(self):
        ''' return full matrix np.exp(-C / beta)'''
        raise NotImplementedError()


def sinkhorn_toeplitz(X, Y, bin_size, beta=0.01, max_iter=200, store_err=True, early_stopping=True,
                      eps=1e-12, tol=1e-10, patience=10, verbose=True, plot=True):
    '''
    X, Y - datapoints of two distributions
    bin_size - number of bins (equal for each dimension)
    beta - regularization parameter 
    '''
    bins, p, q = binning(X, Y, bin_size)
    size = p.shape
    p = p.ravel()
    q = q.ravel()
    top = cdist(bins[0].reshape(1, -1), bins)
    K = Toeplitz(np.exp(- top / beta), size)
    b = np.ones(np.prod(size))
    
    i = 0
    j = 0
    err = [10]
    for i in range(max_iter):
        with np.errstate(divide='ignore', invalid='ignore'):
            Kb = K.matvec(b)
            a = np.nan_to_num(np.divide(p, Kb))
#             a = np.divide(p, Kb)
            Ka = K.matvec(a)
            b = np.nan_to_num(np.divide(q, Ka))
#             b = np.divide(q, Ka)
        if store_err:
#             g = np.diag(a) @ K_full @ np.diag(b)
            g0 = a * (K.matvec(b))
            g1 = b * (K.matvec(a))
#             err.append(np.linalg.norm(g.sum(0) - q) + np.linalg.norm(g.sum(1) - p))
            err.append(np.linalg.norm(g0 - p) + np.linalg.norm(g1 - q))
            #########################################################
            if early_stopping:
                # if good enough
                if err[-1] < eps:
                    if verbose:
                        print(f'#iterations={i+1}, early stopping: eps, err={err[-1]}')
                    break
                # if no improvements
                if np.abs(err[-2] - err[-1]) < tol:
                    j += 1
                    if j > patience:
                        if verbose:
                            print(f'#iterations={i+1}, early stopping: tol, err={err[-1]}')
                        break
                else:
                    j = 0
    else:
        if verbose:
            print(f'#iterations={i},  err={err[-1]}')
    if plot:
        plt.figure(figsize=(10,4))
        if store_err:
            plt.subplot(121)
            plt.title('error')
            plt.semilogy(range(len(err)-1), err[1:])
            plt.subplot(122)
        plt.title('optimal transport matrix')
        K_full = np.exp(-cdist(bins, bins) / beta)
        plt.imshow(a.reshape(-1,1) * K_full * b.reshape(1,-1))
        plt.show()
    
    return K_full, a, b, bins, p, q


# def sinkhorn_toeplitz(X, Y, bin_size, beta=0.01, max_iter=200, stop=True, eps=1e-12, tol=1e-10, patience=10, verbose=True, plot=True):
#     '''
#     X, Y - datapoint of two distributions
#     bin_size - number of bins (equal for each dimension)
#     beta - regularization parameter 
#     '''
#     bins, p, q = binning(X, Y, bin_size)
#     size = p.shape
#     p = p.ravel()
#     q = q.ravel()
#     top = cdist(bins[0].reshape(1,-1), bins)
#     K = Toeplitz(np.exp(- top / beta), size)
#     b = np.ones(np.prod(size))
    
#     i = 0
#     if stop:
#         j = 0
#         err = [10]
#     while i < max_iter:
#         Kb = K.matvec(b)
#         a = np.divide(p, Kb)
#         Ka = K.matvec(a)
#         b = np.divide(q, Ka)
        
#         if stop:
#             g0 = b * (K.matvec(a))
#             g1 = a * (K.matvec(b))

#             err.append(np.linalg.norm(g0 - q) + np.linalg.norm(g1 - p))
#             #########################################################
#             # if good enough
#             if err[-1] < eps:
#                 if verbose:
#                     print(f'#iterations={i+1}, early stopping: eps, err={err[-1]}')
#                 break
#             # if no improvements
#             if np.abs(err[-2] - err[-1]) < tol:
#                 j += 1
#                 if j > patience:
#                     if verbose:
#                         print(f'#iterations={i+1}, early stopping: tol, err={err[-1]}')
#                     break
#             else:
#                 j = 0
#         i +=1
#     if plot:
#         plt.figure(figsize=(10,4))
#         plt.subplot(121)
#         plt.title('error')
#         plt.semilogy(range(len(err)-1), err[1:])
# #         plt.subplot(122)
# #         plt.title('optimal transport matrix')
# #         plt.imshow()
#         plt.show()
    
#     return K, a, b