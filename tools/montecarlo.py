import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib as mpl
import matplotlib.pyplot as plt

from tools.general import Display

class MonteCarloSearch:
    def __init__(self, traj, n=100):
        self.n = n # total number of evaluations
        self.traj = traj
        self.res = np.zeros((n,2)) # stores the coordinates, the destination and the score associated to each evaluation
        self.points = np.zeros((n,2))
        self.opt_ind = np.zeros((2,1))
        self.opt = np.zeros(2)
        self.vbest = np.zeros(2)

    def montecarlo_search(self):
        self.points = self.traj.span*(np.random.random(size=(self.n,2))-0.5)+np.mean(self.traj.bounds,axis=1)
        for i in range(self.n):
            self.res[i] = self.traj.evaluate(self.points[i,:]-self.traj.p0)
        return self.points.copy(), self.res.copy()

    def vbest_montecarlo(self, verbose = True):
        self.opt_indices = np.array(np.where(self.res[:,1] == self.res[:,1].max())[0])
        self.opt = self.opt_indices[np.random.randint(self.opt_indices.shape[0])]
        self.vbest = self.points[self.opt]-self.traj.p0
        if verbose:
            print('{} maximum(s) - proposal [{}] : v = [{:.2f},{:.2f}]'.format(self.opt_indices.shape[0],self.opt,self.vbest[0],self.vbest[1]))
        self.traj.reset(v0=self.vbest)

class DisplayMonteCarlo(Display, MonteCarloSearch):
    def __init__(self,traj,n=100):
        Display.__init__(self, traj)
        MonteCarloSearch.__init__(self, traj, n)
        self.montecarlo_search()
        self.vbest_montecarlo()
        self.points_vor = np.append(self.points, [[self.traj.bounds[0,0]-self.traj.span[0],self.traj.bounds[1,0]-self.traj.span[1]],
                                                  [self.traj.bounds[0,0]-self.traj.span[0],self.traj.bounds[1,1]+self.traj.span[1]],
                                                  [self.traj.bounds[0,1]+self.traj.span[0],self.traj.bounds[1,0]-self.traj.span[1]],
                                                  [self.traj.bounds[0,1]+self.traj.span[0],self.traj.bounds[1,1]+self.traj.span[1]]], axis=0) # to avoid infinite voronoi regions inside of bounds
        self.vor = Voronoi(self.points_vor[:,:2])

    def plot_mc_config(self,title=False,ax=None):
        if ax is None :
            _, ax = plt.subplots(figsize=(32,32),layout='constrained',subplot_kw = {'aspect':1})
        self.plot_config(title,ax)
        voronoi_plot_2d(self.vor, ax, show_vertices=False, line_colors='w',point_size=np.sqrt(ax.get_window_extent().width* ax.get_window_extent().height)/(np.sqrt(self.n)*10),line_alpha=0.5)
        ax.axis([self.traj.bounds[0,0], self.traj.bounds[0,1], self.traj.bounds[1,0], self.traj.bounds[1,1]])
        ax.set_xticks([]),ax.set_yticks([])

    def vor_crashsite(self,ax):
        ax.set_title('Final Destination')
        ax.axis([self.traj.bounds[0,0], self.traj.bounds[0,1], self.traj.bounds[1,0], self.traj.bounds[1,1]])
        ax.set_xticks([]),ax.set_yticks([])
        for i in range(self.n): # no need to add "if not -1 in region :" since those are at n,...,n+4
            polygon = [self.vor.vertices[i] for i in self.vor.regions[self.vor.point_region[i]]]
            ax.fill(*zip(*polygon),color = self.color_list[int(self.res[i,0])-1])
        cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=self.colormap),ax=ax,ticks=(np.arange(self.traj.N+2)+0.5)/(self.traj.N+2))
        cbar.ax.set_yticklabels(self.colorbar_ticklabels,rotation='vertical',verticalalignment='center')
        for i in range(self.traj.N):
            ax.add_patch(plt.Circle((self.traj.loc[i,0], self.traj.loc[i,1]), self.traj.radius,color='k',alpha=0.2))
            ax.text(self.traj.loc[i,0] , self.traj.loc[i,1],'{}'.format(i+1),horizontalalignment='center',verticalalignment='center',c='w',alpha=0.5,size=20)
        ax.scatter((self.traj.p0+self.vbest)[0],(self.traj.p0+self.vbest)[1], s=80, facecolors='none', edgecolors='r',linewidths=2)
        ax.scatter(self.traj.p0[0],self.traj.p0[1],c='gold',marker='+',s=1e-3*min(ax.get_window_extent().width, ax.get_window_extent().height)**2)

    def vor_score(self, ax):
        ax.set_title('Flight time')
        ax.axis([self.traj.bounds[0,0], self.traj.bounds[0,1], self.traj.bounds[1,0], self.traj.bounds[1,1]])
        ax.set_xticks([]),ax.set_yticks([])
        for i in range(self.n): # no need to add if not -1 in region since those are at n,...,n+4
            polygon = [self.vor.vertices[i] for i in self.vor.regions[self.vor.point_region[i]]]
            ax.fill(*zip(*polygon),color = mpl.colormaps['managua']((np.log10(self.res[i,1])-np.log10(self.res[:,1].min()))/(np.log10(self.res[:,1].max())-np.log10(self.res[:,1].min()))))
        cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=mpl.colormaps['managua']),ax=ax,extend='max' if np.max(self.res[...,1].T)==self.traj.Tmax else 'neither')
        cbar.set_ticks(ticks=[(i-np.log10(self.res[:,1].min()))/(np.log10(self.res[:,1].max())-np.log10(self.res[:,1].min())) for i in range(int(np.floor(np.log10(np.max(self.res[...,1]))+1)))])
        cbar.ax.minorticks_on(); cbar.ax.yaxis.set_ticks([(np.log10(j*10**i)-np.log10(self.res[:,1].min()))/(np.log10(self.res[:,1].max())-np.log10(self.res[:,1].min())) 
                                                          for i in range(int(np.floor(np.log10(np.min(self.res[...,1])))),int(np.floor(np.log10(np.max(self.res[...,1]))+1)))
                                                          for j in range(10) if j*10**i>=self.res[:,1].min() and j*10**i<=self.res[:,1].max()], minor=True)
        cbar.ax.set_yticklabels(labels=['$10^{}$'.format(i) for i in range(int(np.floor(np.log10(np.max(self.res[...,1]))+1)))],verticalalignment='center')
        ax.scatter((self.traj.p0+self.vbest)[0],(self.traj.p0+self.vbest)[1], s=80, facecolors='none', edgecolors='r',linewidths=2)

    def plot4(self, figsize = None, save = False):
        fig, axs = plt.subplots(2,2,figsize=(16,16*self.traj.span[1]/self.traj.span[0]) if figsize is None else figsize,layout='constrained',subplot_kw = {'aspect':1})
        self.plot_mc_config(title=True, ax=axs[0,0])
        self.plot_traj(self.vbest,ax=axs[1,0])
        self.vor_crashsite(axs[0,1])
        self.vor_score(axs[1,1])
        if save :
            np.save('save/planets_loc/[{}, {}].npy'.format(self.traj.p0[0],self.traj.p0[1]),self.traj.loc)
            np.save('save/montecarlo/[{}, {}].npy'.format(self.traj.p0[0],self.traj.p0[1]),self.points)
            fig.savefig('figs/[{}, {}] - mc({}).png'.format(self.traj.p0[0],self.traj.p0[1],self.traj.N,self.n),bbox_inches='tight')