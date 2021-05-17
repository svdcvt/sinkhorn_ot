import mkl_fft
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.spatial.distance import cdist
from tqdm import tqdm
import IPython.display
from time import time


def sinkhorn(X, Y, beta=0.01, max_iter=200, store_err=True, early_stopping=True, 
             eps=1e-12, tol=1e-10, patience=10, verbose=True, plot_err=False, plot_mat=False):
    '''
    X, Y - datapoint of two distributions
    beta - regularization parameter 
    
    return: K, a, b
    '''
    n, m = len(X), len(Y)
    C = cdist(X, Y)
    # C /= C.max()
    
    p = np.ones(n) / n
    q = np.ones(m) / m
    
    K = np.exp(-C / beta)
    Kt = K.T
    b = np.ones(m)
    
    i = 0
    j = 0
    err = [10]
    t = []
    for i in range(max_iter):
        with np.errstate(divide='ignore', invalid='ignore'):
            s = time()
            Kb = K.dot(b)
            a = np.divide(p, Kb)
            Kta = Kt.dot(a)
            b = np.divide(q, Kta)
            t.append(time() - s)
        if store_err:
            g0 = a * K.dot(b)
            g1 = b * Kt.dot(a)
            err.append(np.linalg.norm(g0 - p) + np.linalg.norm(g1 - q))
            ###########################################################
            if verbose == 2:
                print(f'{i:5.0f}: {err[-1]:.20f}')
            if plot_err == 2:
                    IPython.display.clear_output(wait=True)
                    plt.figure(figsize=(10,4))
                    plt.title(f'error rate, {np.mean(t)*1000:3.3f}ms per iteration')
                    plt.semilogy(range(len(err)-1), err[1:])
                    plt.show()
            
            if early_stopping:
                # if good enough
                if err[-1] < eps:
                    if verbose:
                        print(f'#iterations={i+1}, early stopping: eps, err={err[-1]:.5e}, {np.mean(t)*1000:3.3f}ms per iteration')
                    break
                # if no improvements
                if np.abs(err[-2] - err[-1]) < tol:
                    j += 1
                    if j > patience:
                        if verbose:
                            print(f'#iterations={i+1}, early stopping: tol, err={err[-1]:.5e}, {np.mean(t)*1000:3.3f}ms per iteration')
                        break
                else:
                    j = 0
    else:
        if verbose:
            print(f'#iterations={i+1},  err={err[-1]:.5e}, {np.mean(t)*1000:3.3f}ms per iteration')
    print('Finished')

    if store_err and plot_err == 1:
        plt.figure(figsize=(10,4))
        plt.subplot(121)
        plt.title('error')
        plt.semilogy(range(len(err)-1), err[1:])
        if plot_mat: plt.subplot(122)
    elif plot_mat:
        plt.figure(figsize=(5,4))
    if plot_mat:
        plt.title('optimal transport matrix')
        plt.imshow(a.reshape(-1,1) * K * b.reshape(1,-1))
    if plot_mat or plot_err: plt.show()
    method='sinkhorn'
    np.save(f'K_{method}.npy', K)
    np.save(f'a_{method}.npy', a)
    np.save(f'b_{method}.npy', b)
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
    
    grid = np.linspace(np.min(clouds, 0), np.max(clouds, 0), bin_size + 1).T # [Dimension, Bins]
    
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
        
    def matvec(self, x, debug=True):
        ''' fast "matvec" multiplication '''
        if x.ndim > 1:
            if (x.shape[0] == 1 or x.shape[1] == 1):
                x = x.ravel()
            else:
                raise ValueError()
        x_fft = mkl_fft.fftn(np.pad(x.reshape(self.size), self.pad))
        if debug:
            print('*'*100)
            print('circ_fft\n', self.circ_fft)
            print('x_fft\n', x_fft)
            print('multiplication\n', self.circ_fft * x_fft)
            print('ifft\n', mkl_fft.ifftn(self.circ_fft * x_fft))
            print('result\n', np.abs(mkl_fft.ifftn(np.multiply(self.circ_fft, x_fft))[self.area]).ravel())
            print('*'*100)
        return np.real(mkl_fft.ifftn(np.multiply(self.circ_fft, x_fft))[self.area]).ravel()
    
    def full(self):
        ''' return full matrix np.exp(-C / beta) without recomputation'''
        raise NotImplementedError()


def sinkhorn_toeplitz(X, Y, bin_size, beta=0.01, max_iter=200,
                      warm_start=None,
                      early_stopping=True, eps=1e-12, tol=1e-10, patience=10, 
                      verbose=True, store_full=False, store_err=True, plot=0, debug=False, debug_=False):
    '''
    Arguments
    
    X, Y:
        ndarray:
            if bin_size is int: datapoints of two distributions
            if bin_size is ndarray: mass distibutions over bins for two distributions
    bin_size:
        int: number of bins (equal for each dimension) foo binning,
        ndarray: array of size [N, D] with coordinates of bin centers
    beta:
        float: entropy regularization parameter 
    early_stopping:
        bool: whether to stop iterations if the error $E(\gamma)$ value 
              matches the conditions (eps, tol, patience):
            $E(\gamma) = ||\gamma @ 1 - p||_2 + ||\gamma^T @ 1 - q||_2
    eps:
        float: stops iterations if $E(\gamma)$ < eps
    tol:
        float: stops iterations if $E(\gamma)_i - $E(\gamma)_{i-1}$ < tol
    patience:
        int: 
    verbose:
        0, 1, 2 (or bool): controls the verbosity:
            0 (False): nothing to be printed
            1 (True): print stopping criteria or/and final error
            2: print every iteration error
    store_full:
        bool: whether to calculate and return full matrix K = exp(-C/beta)
    store_err:
        bool: whether to calculate $E(\gamma)$ (if False, then early_stopping is ignored)
    plot:
        0, 1, 2 (or bool): whether to plot error linegraph (if store_err is True)
            0 (False): nothing to be plotted
            1 (True): one plot after finished iteration process
            2: real-time plot during iteration process
    
    Return
    K: 
        Toeplitz or ndarray
    a, b, bins, p, q:
        ndarrays
    
    '''
    if isinstance(bin_size, int):
        bins, p, q = binning(X, Y, bin_size)
    elif isinstance(bin_size, np.ndarray):
        bins, p, q = bin_size.copy(), X.copy(), Y.copy()
        p /= p.sum()
        q /= q.sum()
    else:
        raise ValueError()
    
    size = p.shape
    p = p.ravel() + 1e-15
    q = q.ravel() + 1e-15
    top = cdist(bins[0].reshape(1, -1), bins) # O(B)
    # top /= top.max()
    K = Toeplitz(np.exp(- top / beta), size)
    if debug_:
        C = cdist(bins, bins)
        K_ = np.exp(-C / beta)
    if warm_start is None:
        b = np.ones(np.prod(size))
    else:
        a, b = warm_start
    
    i = 0
    j = 0
    err = [10]
    t = []
    print('Starting iterative process')
    
    for i in range(max_iter):
        
        with np.errstate(divide='raise', over='raise', under='raise'):
            try:
                if debug or debug_:
                    print(f'ITERATION {i}')
                    print('-'*100)
                s = time()
                if debug: print('MATVEC by b\n b =\n', b)
                Kb = K.matvec(b, debug) # O(B log B)
                if debug_:
                    K_b = K_ @ b
                    print('.'*100)
                    print("\nISCLOSE TO REAL MATVEC:", np.all(np.isclose(K_b, Kb, 1e-10)), 
                          '\n\tabs(diff).mean', np.abs(K_b - Kb).mean(),
                          '\n\tabs(diff).mean / abs(K_b).mean', np.abs(K_b - Kb).mean() / np.abs(K_b).mean())
                    print('.'*100)
                
                if debug: print(f'DIVIDING p by K.b\n p =\n{p}\n K.b =\n', Kb)
                a = np.divide(p, Kb)
                if debug: print('MATVEC by a\n a =\n', a)
                Ka = K.matvec(a, debug) # O(B log B)
                          
                if debug_:
                    K_a = K_ @ a
                    print('.'*100)
                    print("\nISCLOSE TO REAL MATVEC:", np.all(np.isclose(K_a, Ka, 1e-10)), 
                          '\n\tabs(diff).mean', np.abs(K_a - Ka).mean(),
                          '\n\tabs(diff).mean / abs(K_a).mean', np.abs(K_a - Ka).mean() / np.abs(K_a).mean())
                    print('.'*100)
                          
                if debug: print(f'DIVIDING q by K.a\n q =\n{q}\n K.a =\n', Ka) 
                b = np.divide(q, Ka)
                if debug: print('-'*100)
                t.append(time() - s)
            except FloatingPointError as e:
                if verbose:
                    print(e)
                    print(f'#iterations={i+1},  err={err[-1]:.5e}, {np.mean(t)*1000:3.3f}ms per iteration')
                break
        

        if store_err:
            with np.errstate(divide='raise', over='raise', under='raise'):
                try:
                    g0 = a * (K.matvec(b, debug))
                    g1 = b * (K.matvec(a, debug))
                except:
                    pass
                err.append(np.linalg.norm(g0 - p) + np.linalg.norm(g1 - q))
            ###########################################################
            if verbose == 2:
                print(f'{i:5.0f}: {err[-1]:.20f}')
            if plot == 2:
                    IPython.display.clear_output(wait=True)
                    plt.figure(figsize=(10,4))
                    plt.title(f'error rate, {np.mean(t)*1000:3.3f}ms per iteration')
                    plt.semilogy(range(len(err)-1), err[1:])
                    plt.savefig('error.png')
                    plt.show()
            
            if early_stopping:
                # if good enough
                if err[-1] < eps:
                    if verbose:
                        print(f'#iterations={i+1}, early stopping: eps, err={err[-1]:.5e}, {np.mean(t)*1000:3.3f}ms per iteration')
                    break
                # if no improvements
                if np.abs(err[-2] - err[-1]) < tol:
                    j += 1
                    if j > patience:
                        if verbose:
                            print(f'#iterations={i+1}, early stopping: tol, err={err[-1]:.5e}, {np.mean(t)*1000:3.3f}ms per iteration')
                        break
                else:
                    j = 0
                # if error goes up
                if err[-1] > err[-2]:
                    if verbose:
                        print(f'#iterations={i+1}, early stopping: error up, err={err[-1]:.5e}, {np.mean(t)*1000:3.3f}ms per iteration')
                    break
                
    else:
        if verbose:
            print(f'#iterations={i+1},  err={err[-1]:.5e}, {np.mean(t)*1000:3.3f}ms per iteration')
    print('Finished')
    if store_full:
        K_full = np.exp(-cdist(bins, bins) / beta)
    if plot:
        if plot == 1 or store_full:
            plt.figure(figsize=(10,4))
        if store_err and plot == 1:
            plt.subplot(1,2,1)
            plt.title('error')
            plt.semilogy(range(len(err)-1), err[1:])
        if store_full:
            if plot: plt.subplot(1,2,3-plot)
            plt.title('optimal transport matrix')
            K_full = np.exp(-cdist(bins, bins) / beta)
            plt.imshow(a.reshape(-1,1) * K_full * b.reshape(1,-1))
        plt.savefig('error.png')
        plt.show()
    if store_full:
        return K_full, a, b, bins, p, q
    else:
        method='toeplitz'
        np.save(f'K_{method}.npy', K)
        np.save(f'a_{method}.npy', a)
        np.save(f'b_{method}.npy', b)
        return K, a, b, bins, p, q
