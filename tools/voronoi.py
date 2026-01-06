import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib as mpl
import matplotlib.pyplot as plt

from tools.general import Display

class MonteCarloSearch:
    def __init__(self, traj, n=100):
        self.n = n # total number of evaluations
        self.traj = traj
        self.points = np.zeros((n,2)) # coordinates at which the score is evaluated
        self.res = np.zeros((n,2)) # results of each evaluation (destination and score)
        self.opt_ind = np.zeros((2,1)) # list of optimum indices
        self.opt = np.zeros(2) # an element from opt_ind
        self.vbest = np.zeros(2) # initial velocity for optimum
        self.montecarlo_search()
        self.vbest_montecarlo()

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

class FractalSearch:
    def __init__(self, traj, n=100):
        self.n = n
        self.traj = traj
        self.points = np.zeros((n,2)) # coordinates at which the score is evaluated
        self.res = np.zeros((n,3)) # results of each evaluation (depth, destination and score)
        self.L = [] # list of segments to explore, containing (index of point1, index of point2, best score among point1 and point2)
        self.opt_ind = np.zeros((2,1)) # list of optimum indices
        self.opt = np.zeros(2) # an element from opt_ind
        self.vbest = np.zeros(2) # initial velocity for optimum

    def fractal_search_crosspattern(self,lvl0=0,lvlmax=8,greed=False):
        self.points[:2] = np.array([[self.traj.bounds[0,0],self.traj.bounds[1,0]], [self.traj.bounds[1,1],self.traj.bounds[1,1]]],dtype=float)
        self.res[0,1:], self.res[1,1:] = self.traj.evaluate(self.points[0]-self.traj.p0), self.traj.evaluate(self.points[1]-self.traj.p0)
        n_call = 2
        self.L = [(0,1,max(self.res[0,-1],self.res[1,-1]))]
        L_rejected = []
        while n_call<self.n-2 and len(self.L)>0:
            s = np.zeros((5),dtype=int)
            s[0], s[1], _ = self.L.pop()
            lvl = max(self.res[s[0],0],self.res[s[1],0])
            if len(self.L)==0 :
                if greed == True:
                    L_rejected.sort(key=lambda x:x[-1])
                self.L = L_rejected # si on a rien trouvé de mieux, autant aller chercher des frontières qu'on aurait pu rater entre les mailles
            mid, point1, point2 = (self.points[s[0]]+self.points[s[1]])/2, np.array([self.points[s[0],0],self.points[s[1],1]]), np.array([self.points[s[1],0],self.points[s[0],1]])
            #point1, point2 = mid + np.array([[0,-1],[1,0]]).dot(self.points[s[0]]-mid), mid - np.array([[0,-1],[1,0]]).dot(self.points[s[0]]-mid) #true rotation
            if np.where(np.all(self.points[:n_call] == mid,axis=1))[0].size == 0:
                s[4] = n_call
                self.points[s[4]] = mid
                self.res[s[4]] = np.concatenate([[lvl+1],self.traj.evaluate(mid-self.traj.p0)])
                n_call+=1
            else :
                s[4] = np.where(np.all(self.points == mid,axis=1))[0][0]
            # check whether the point has already been evaluated (otherwise many evaluations get wasted on the same point)
            if np.where(np.all(self.points[:n_call] == point1,axis=1))[0].size == 0:
                s[2] = n_call
                self.points[s[2]] = point1
                self.res[s[2]] = np.concatenate([[lvl+1], self.traj.evaluate(point1-self.traj.p0)])
                n_call+=1
            else :
                s[2] = np.where(np.all(self.points[:n_call] == point1,axis=1))[0][0]
            if np.where(np.all(self.points == point2,axis=1))[0].size == 0:
                s[3] = n_call
                self.points[s[3]] = point2
                self.res[s[3]] = np.concatenate([[lvl+1], self.traj.evaluate(point2-self.traj.p0)])
                n_call+=1
            else :
                s[3] = np.where(np.all(self.points == point2,axis=1))[0][0]
            # list of new candidates (accepted an rejected)
            L_new = [(s[i],s[4],max(self.res[s[i],-1],self.res[s[4],-1])) for i in range(4) if (self.res[s[i],1] != self.res[s[4], 1] or self.res[s[i],0]<lvl0) and lvl<lvlmax-1]
            L_newrejected = [(s[i],s[4],max(self.res[s[i],-1],self.res[s[4],-1])) for i in range(4) if self.res[s[i],1] == self.res[s[4], 1] and self.res[s[i],0]>=lvl0]
            L_new.sort(key=lambda x:x[-1]); L_newrejected.sort(key=lambda x:x[-1])
            self.L+=L_new; L_rejected = L_newrejected + L_rejected
            if lvl<lvl0:
                self.L.sort(key=lambda x:(-max(self.res[x[0],0],self.res[x[1],0]),-x[-1])) # width search starting with bad solutions to save the best for last and put them on top of the pile
            elif greed == True: # actually just need to put the max on top, not sort the whole list at each addition
                self.L.sort(key=lambda x:(x[-1])) # greedy search, for width search see 2 lines above, otherwise you dive depth-first, faster with bisect.insort or even better heapq.merge
        # on pourrait bourrer les 2 dernières évaluations si L non vide mais la méthode est pas assez intéressante pour se prendre la tête, je veux juste pas de 0 pour pas casser la logcolorbar
        self.points[n_call:], self.res[n_call:] = np.tile(self.points[0],self.n-n_call).reshape(self.n-n_call,2), np.tile(self.res[0],self.n-n_call).reshape(self.n-n_call,self.res.shape[1]) 


    def delaunay_middles(self):
        # les milieux des segments dans la triangulation de Delaunay où les deux extrémités sont différentes
        return
    
    def voronoi_edges(self):
        # les sommets de Voronoï dont au moins deux des trois points autour sont différends
        return
        

    def get_vbest(self, verbose = True):
        self.opt_indices = np.array(np.where(self.res[:,-1] == self.res[:,-1].max())[0])
        self.opt = self.opt_indices[np.random.randint(self.opt_indices.shape[0])]
        self.vbest = self.points[self.opt]-self.traj.p0
        if verbose:
            print('{} maximum(s) - proposal [{}] : v = [{:.2f},{:.2f}]'.format(self.opt_indices.shape[0],self.opt,self.vbest[0],self.vbest[1]))
        self.traj.reset(v0=self.vbest)


class DisplayVoronoi(Display):
    def __init__(self,traj,solver):
        Display.__init__(self, traj)
        self.sol = solver
        self.points_vor = np.append(self.sol.points, [[self.traj.bounds[0,0]-self.traj.span[0],self.traj.bounds[1,0]-self.traj.span[1]],
                                                      [self.traj.bounds[0,0]-self.traj.span[0],self.traj.bounds[1,1]+self.traj.span[1]],
                                                      [self.traj.bounds[0,1]+self.traj.span[0],self.traj.bounds[1,0]-self.traj.span[1]],
                                                      [self.traj.bounds[0,1]+self.traj.span[0],self.traj.bounds[1,1]+self.traj.span[1]]], axis=0) # to avoid infinite voronoi regions inside of bounds
        self.vor = Voronoi(self.points_vor[:,:2])

    def plot_vor_config(self,title=False,ax=None):
        if ax is None :
            _, ax = plt.subplots(figsize=(32,32),layout='constrained',subplot_kw = {'aspect':1})
        self.plot_config(title,ax)
        voronoi_plot_2d(self.vor, ax, show_vertices=False, line_colors='w',point_size=np.sqrt(ax.get_window_extent().width* ax.get_window_extent().height)/(np.sqrt(self.sol.n)*10),line_alpha=0.5)
        ax.axis([self.traj.bounds[0,0], self.traj.bounds[0,1], self.traj.bounds[1,0], self.traj.bounds[1,1]])
        ax.set_xticks([]),ax.set_yticks([])

    def vor_crashsite(self,ax):
        ax.set_title('Final Destination')
        ax.axis([self.traj.bounds[0,0], self.traj.bounds[0,1], self.traj.bounds[1,0], self.traj.bounds[1,1]])
        ax.set_xticks([]),ax.set_yticks([])
        for i in range(self.sol.n): # no need to add "if not -1 in region :" since those are at n,...,n+4
            polygon = [self.vor.vertices[i] for i in self.vor.regions[self.vor.point_region[i]]]
            ax.fill(*zip(*polygon),color = self.color_list[int(self.sol.res[i,1])-1],ec='w',lw=0.2)
        cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=self.colormap),ax=ax,ticks=(np.arange(self.traj.N+2)+0.5)/(self.traj.N+2))
        cbar.ax.set_yticklabels(self.colorbar_ticklabels,rotation='vertical',verticalalignment='center')
        for i in range(self.traj.N):
            ax.add_patch(plt.Circle((self.traj.loc[i,0], self.traj.loc[i,1]), self.traj.radius,color='k',alpha=0.2))
            ax.text(self.traj.loc[i,0] , self.traj.loc[i,1],'{}'.format(i+1),horizontalalignment='center',verticalalignment='center',c='w',alpha=0.5,size=20)
        ax.scatter((self.traj.p0+self.sol.vbest)[0],(self.traj.p0+self.sol.vbest)[1], s=80, facecolors='none', edgecolors='r',linewidths=2)
        ax.scatter(self.traj.p0[0],self.traj.p0[1],c='gold',marker='+',s=1e-3*min(ax.get_window_extent().width, ax.get_window_extent().height)**2)

    def vor_score(self, ax):
        ax.set_title('Flight time')
        ax.axis([self.traj.bounds[0,0], self.traj.bounds[0,1], self.traj.bounds[1,0], self.traj.bounds[1,1]])
        ax.set_xticks([]),ax.set_yticks([])
        for i in range(self.sol.n): # no need to add if not -1 in region since those are at n,...,n+4
            polygon = [self.vor.vertices[i] for i in self.vor.regions[self.vor.point_region[i]]]
            ax.fill(*zip(*polygon),color = mpl.colormaps['managua']((np.log10(self.sol.res[i,-1])-np.log10(self.sol.res[:,-1].min()))/(np.log10(self.sol.res[:,-1].max())-np.log10(self.sol.res[:,-1].min()))),ls='',lw=0)
        # Getting a Voronoi colorbar to behave just as it would for a regular heatmap
        cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=mpl.colormaps['managua']),ax=ax,extend='max' if np.max(self.sol.res[...,-1].T)==self.traj.Tmax else 'neither')
        cbar.set_ticks(ticks=[(i-np.log10(self.sol.res[:,-1].min()))/(np.log10(self.sol.res[:,-1].max())-np.log10(self.sol.res[:,-1].min())) for i in range(int(np.floor(np.log10(np.max(self.sol.res[...,-1]))+1)))])
        cbar.ax.minorticks_on(); cbar.ax.yaxis.set_ticks([(np.log10(j*10**i)-np.log10(self.sol.res[:,-1].min()))/(np.log10(self.sol.res[:,-1].max())-np.log10(self.sol.res[:,-1].min())) 
                                                          for i in range(int(np.floor(np.log10(np.min(self.sol.res[...,-1])))),int(np.floor(np.log10(np.max(self.sol.res[...,-1]))+1)))
                                                          for j in range(10) if j*10**i>=self.sol.res[:,-1].min() and j*10**i<=self.sol.res[:,-1].max()], minor=True)
        cbar.ax.set_yticklabels(labels=['$10^{}$'.format(i) for i in range(int(np.floor(np.log10(np.max(self.sol.res[...,-1]))+1)))],verticalalignment='center')
        ax.scatter((self.traj.p0+self.sol.vbest)[0],(self.traj.p0+self.sol.vbest)[1], s=80, facecolors='none', edgecolors='r',linewidths=2)

    def plot4(self, figsize = None, save = False):
        fig, axs = plt.subplots(2,2,figsize=(16,16*self.traj.span[1]/self.traj.span[0]) if figsize is None else figsize,layout='constrained',subplot_kw = {'aspect':1})
        self.plot_vor_config(title=True, ax=axs[0,0])
        self.plot_traj(self.sol.vbest,ax=axs[1,0])
        self.vor_crashsite(axs[0,1])
        self.vor_score(axs[1,1])
        if save :
            np.save('save/planets_loc/[{}, {}].npy'.format(self.traj.p0[0],self.traj.p0[1]),self.traj.loc)
            np.save('save/voronoi/[{}, {}].npy'.format(self.traj.p0[0],self.traj.p0[1]),self.sol.points)
            fig.savefig('figs/[{}, {}] - vor({}).png'.format(self.traj.p0[0],self.traj.p0[1],self.sol.n),bbox_inches='tight')