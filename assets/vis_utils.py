import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np

def visualize_grid(W, ubound=1, padding=1):
    padding = 1
    n_feats = W.shape[0]
    n_inputs = W.shape[1]
    W_full = np.zeros((W.shape[2],W.shape[3]*n_feats + (n_feats-1)*padding,W.shape[4]*n_inputs + (n_inputs-1)*padding))
    for i in range(n_feats):
        for j in range(n_inputs):
            W_full[:,i*(padding+W.shape[3]):(i+1)*(padding+W.shape[3])-1,j*(padding+W.shape[4]):(j+1)*(padding+W.shape[4])-1] = W[i,j,:,:,:]
    return visualize_matrix(W_full,ubound=ubound)

def visualize_matrix(X,interval_len=500,ubound = 1):
    fig = plt.figure()
    ax = plt.gca()
    im = plt.imshow(X[0,:,:], cmap='Greys_r', vmin=-ubound, vmax=ubound,interpolation='none')
    def up(t):
        im.set_array(X[t,:,:])
        return im,
    ani = anim.FuncAnimation(fig, up, xrange(X.shape[0]), interval=interval_len, blit=True, repeat_delay=1000)
    plt.show()
    return ani

def feature_inversion(model):
    pass