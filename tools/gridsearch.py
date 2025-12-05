import numpy as np
import matplotlib.pyplot as plt

from tools.general import Display

class GridSearch :
    def __init__(self,traj,n,bounds=None):
        self.traj = traj # instance of Trajectory
        self.ngrid = n # grid resolution
        self.bounds = bounds # boundaries of solution exploration
        if bounds is None:
            self.bounds = self.traj.bounds
        
        self.gridx, self.gridy = np.linspace(self.bounds[0,0], self.bounds[0,1], self.ngrid), np.linspace(self.bounds[1,0], self.bounds[1, 1], self.ngrid)
        self.gridxx, self.gridyy = np.meshgrid(self.gridx,self.gridy)
        self.gridxy = np.stack([np.repeat(self.gridx,self.gridy.size).reshape((self.gridx.size,self.gridy.size)),np.tile(self.gridy,self.gridx.size).reshape((self.gridx.size,self.gridy.size))],axis=-1)
        
        self.res = np.zeros((self.gridx.size,self.gridy.size,2))

        self.opt_ind = np.zeros((2,1))
        self.opt = np.zeros(2)
        self.vbest = np.zeros(2)
        
    def compute_grid(self, border=True):
        for i in range(self.gridx.size):
            for j in range(self.gridy.size):
                self.res[i,j,:] = self.traj.evaluate(np.array([self.gridx[i],self.gridy[j]])-self.traj.p0)

    def vbest_grid(self):
        self.opt_indices = np.array(np.where(self.res[...,1] == self.res[...,1].max())).T
        self.opt = self.opt_indices[np.random.randint(self.opt_indices.shape[0])]
        self.vbest = self.gridxy[self.opt[0],self.opt[1]] - self.traj.p0
        print('{} maximum(s) - proposal [{}, {}] : v = [{:.2f},{:.2f}]'.format(self.opt_indices.shape[0],self.opt[0],self.opt[1],self.vbest[0],self.vbest[1]))
        self.traj.reset(v0=self.vbest)

class DisplayGridSearch(Display, GridSearch):
    def __init__(self, traj, n, bounds = None):
        Display.__init__(self, traj)
        GridSearch.__init__(self, traj, n, bounds)
        self.compute_grid()
        self.vbest_grid()

    def heatmap_crashsite(self, ax, fig):
        ax.set_title('Final destination')
        pcm = ax.pcolormesh(self.gridxx, self.gridyy, self.res[...,0].T, cmap = self.colormap, vmin=0.5, vmax=self.traj.N+2.5)
        ax.axis([self.gridxx.min(), self.gridxx.max(), self.gridyy.min(), self.gridyy.max()])
        ax.set_xticks([]),ax.set_yticks([])
        cbar = fig.colorbar(pcm,ticks = np.arange(self.traj.N+2)+1)
        cbar.ax.set_yticklabels(self.colorbar_ticklabels,rotation='vertical',verticalalignment='center')
        for i in range(self.traj.N):
            ax.add_patch(plt.Circle((self.traj.loc[i,0], self.traj.loc[i,1]), self.traj.radius,color='k',alpha=0.2))
            ax.text(self.traj.loc[i,0] , self.traj.loc[i,1],'{}'.format(i+1),horizontalalignment='center',verticalalignment='center',c='w',alpha=0.5,size=20)
        ax.scatter((self.traj.p0+self.vbest)[0],(self.traj.p0+self.vbest)[1], s=80, facecolors='none', edgecolors='r',linewidths=2)
        ax.scatter(self.traj.p0[0],self.traj.p0[1],c='gold',marker='+',s=100)        

    def heatmap_score(self, ax, fig):
        ax.set_title('Flight time')
        pcm2 = ax.pcolormesh(self.gridxx, self.gridyy, self.res[...,1].T,cmap='managua', norm='log')
        ax.scatter((self.traj.p0+self.vbest)[0],(self.traj.p0+self.vbest)[1], s=80, facecolors='none', edgecolors='r',linewidths=2)
        ax.axis([self.gridxx.min(), self.gridxx.max(), self.gridyy.min(), self.gridyy.max()])
        ax.set_xticks([]),ax.set_yticks([])
        if np.max(self.res[...,1].T)==self.traj.Tmax:
            cbar = fig.colorbar(pcm2,extend='max')
        else :
            cbar = fig.colorbar(pcm2)
        cbar.set_ticks(ticks=[10**i for i in range(int(np.floor(np.log10(np.max(self.res[...,1]))+1)))]); cbar.ax.yaxis.set_minor_formatter(plt.NullFormatter())

    def plot4(self, figsize = (16,16), title = None, save = False):
        fig, axs = plt.subplots(2, 2, figsize=figsize,layout='constrained',subplot_kw = {'aspect':1})
        if not title is None:
            fig.suptitle(title)
        self.plot_config(title=True, ax=axs[0,0])
        self.plot_traj(self.vbest,axs[1,0])
        self.heatmap_crashsite(axs[0,1],fig)
        self.heatmap_score(axs[1,1],fig)
        if save :
            np.save('save/planets_loc/[{}, {}].npy'.format(self.traj.p0[0],self.traj.p0[1]),self.traj.loc)
            title = '[{}, {}] - g({})'.format(self.traj.p0[0],self.traj.p0[1],self.ngrid) if title is None else title
            fig.savefig('figs/{}.png'.format(title),bbox_inches='tight')