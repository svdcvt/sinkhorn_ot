import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def plot_image(target, image, path, method, shape, binsize, beta, limits, every=1, **kwargs):
    
    s = kwargs.get('s', 2)
    alpha = kwargs.get('alpha', 0.5)
    figsize = kwargs.get('figsize', (15,15))
    
    plt.figure(figsize = figsize)
    plt.title(f'Image of source shape and target, binsize={binsize}$^2$, beta={beta}, {method}',
                 fontsize=18)
    plt.scatter(target[::every,0], target[::every,1], s=s, alpha=alpha, c='orange', label='tagret')
    plt.scatter(image[::every,0], image[::every,1], s=s, alpha=alpha, c='green', label='source corresponding image')
    
    lim_min, lim_max = limits if limits is not None else (target.min(-2), target.max(-2))
    plt.legend(fontsize=14, loc='upper right', bbox_to_anchor=(1., 0.98))
    plt.xlabel('$x_1$', fontsize=14)
    plt.xticks(fontsize=14)
    plt.xlim(lim_min[0] - 0.05, lim_max[0] + 0.05)
    plt.ylabel('$x_2$', fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(lim_min[1] - 0.05, lim_max[1] + 0.05)
    # plt.savefig('current_result.png')
    plt.savefig(os.path.join(path, f'image_{method}_{shape}_{binsize}_{beta}.pdf'))
    plt.show()


def plot_map(source, image, path, method, shape, binsize, beta, limits, **kwargs):
    
    head_width = kwargs.get('head_width', 0.01)
    alpha = kwargs.get('alpha', 0.3)
    figsize = kwargs.get('figsize', (15,15))
    
    plt.figure(figsize = figsize)
    plt.title(f'Map from source shape to image, binsize={binsize}$^2$, beta={beta}, {method}',
                 fontsize=18)
    for j in range(len(xbins)):
        plt.arrow(xbins[j, 0], xbins[j, 1], hat_xbins[j, 0] - xbins[j, 0], hat_xbins[j, 1] - xbins[j, 1], 
                  head_width=head_width, alpha=alpha, length_includes_head=True)
    
    lim_min, lim_max = limits if limits is not None else (np.min(np.vstack([source.min(-2), image.min(-2)]), -2),\
                                                          np.max(np.vstack([source.max(-2), image.max(-2)]), -2))
    plt.xlabel('$x_1$', fontsize=14)
    plt.xticks(fontsize=14)
    plt.xlim(lim_min[0] - 0.05, lim_max[0] + 0.05)
    plt.ylabel('$x_2$', fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(lim_min[1] - 0.05, lim_max[1] + 0.05)
    
    # plt.savefig(os.path.join(path, 'current_result.png'))
    plt.savefig(os.path.join(path, f'map_{method}_{shape}_{binsize}_{beta}.pdf'))
    plt.show()

def plot_mesh(points, distribution, path, method, shape, binsize, beta, limits, every=1, **kwargs):
    
    markersize = kwargs.get('markersize', 0)
    linewidth = kwargs.get('linewidth', 0.2)
    figsize = kwargs.get('figsize', (15,15))
    A = kwargs.get('A', 0.7)
    B = kwargs.get('B', 0.3)
                                 
    mask = distribution.reshape(binsize, binsize)
    maskT = mask.T
    horizontals = points.reshape(binsize, binsize, 2)
    verticals = np.transpose(horizontals, (1, 0, 2))
    
    plt.figure(figsize=figsize)
    plt.title(f'Image mesh of {shape} {"($C_1$={A}, $C_2$={B})" if shape == "flower" else ""}, binsize=${binsize}^2$, mesh=${binsize//every}^2$, beta={beta}, {method}',
             fontsize=14)
    
    for i in range(0, binsize, every):
        m1, m2 = mask[i] > 0, maskT[i] > 0
        plt.plot(horizontals[i][m1][::every,0], horizontals[i][m1][::every, 1], 
                 'o-', c='k', linewidth=linewidth, markersize=markersize)
        plt.plot(verticals[i][m2][::every,0], verticals[i][m2][::every, 1], 
                 'o-', c='k', linewidth=linewidth, markersize=markersize)
    
    lim_min, lim_max = limits if limits is not None else (points.min(-2), points.max(-2))
    plt.xlabel('$x_1$', fontsize=14)
    plt.xticks(fontsize=14)
    plt.xlim(lim_min[0] - 0.05, lim_max[0] + 0.05)
    plt.ylabel('$x_2$', fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(lim_min[1] - 0.05, lim_max[1] + 0.05)
    
    # plt.savefig(os.path.join(path, 'current_result.png'))
    plt.savefig(os.path.join(path, f'mesh_{method}_{shape}_{binsize}_{beta}.pdf'))
    plt.show()


def plot_3d_mesh(points, distribution, shape, binsize, path, limits, every=1, animate=False):
    lim_min, lim_max = limits if limits is not None else ((-1, -1, -1), (1, 1, 1))
    plt.figure(figsize=(10,10))
    ax = plt.axes(projection='3d')

    distribution = distribution.reshape(binsize, binsize, binsize)
    bins = bins.reshape(binsize, binsize, binsize, 3)

    zbins = np.transpose(bins, (0, 1, 2, 3)) # fix (x, y), connect z
    ybins = np.transpose(bins, (0, 2, 1, 3)) # fix (x, z), connect y
    xbins = np.transpose(bins, (2, 1, 0, 3)) # fix (z, y), connect x

    xp = np.transpose(p, (0, 1, 2))
    yp = np.transpose(p, (0, 2, 1))
    zp = np.transpose(p, (2, 1, 0))
    
    w = 0.5
    for i in tqdm(list(range(0, binsize, every)) + [binsize-1]):
        for j in list(range(0, binsize, every)) + [binsize-1]:
            if (i == int(every * ((binsize//every)/2)) and ((j == 0) or (j==(binsize-1)))) or\
            (j == int(every * ((binsize//every)/2)) and ((i == 0) or (i==(binsize-1)))) or \
            ((i == 0 or i == (binsize - 1)) and ((j == 0) or (j == (binsize-1)))):
                w = 5
            ax.plot3D(zbins[i][j][:, 0],
                      zbins[i][j][:, 1],
                      zbins[i][j][:, 2], 
                      linewidth=w, c='k')

            ax.plot3D(ybins[i][j][:, 0],
                      ybins[i][j][:, 1],
                      ybins[i][j][:, 2], 
                      linewidth=w, c='k')

            ax.plot3D(xbins[i][j][:, 0],
                      xbins[i][j][:, 1],
                      xbins[i][j][:, 2], 
                      linewidth=w, c='k')
            w = 0.5
    ax.set_xlim(lim_min[0] - 0.05, lim_max[0] + 0.05)
    ax.set_ylim(lim_min[1] - 0.05, lim_max[1] + 0.05)
    ax.set_zlim(lim_min[2] - 0.05, lim_max[2] + 0.05)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    
    if animate:
        for ii in range(0,360,10):
            ax.view_init(elev=30., azim=ii)
            plt.savefig(os.path.join(path, f'animations/{shape}/movie{ii}.png'))
    else:
        # plt.savefig(os.path.join(path, 'current_result.png'))
        plt.savefig(os.path.join(path, f'mesh_{shape}_{binsize}_{beta}.pdf'))
    plt.show()
