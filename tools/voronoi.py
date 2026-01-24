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

    def fractal_grid(self,lvl0=0,lvlmax=8,greed=False):
        self.points, self.res = np.zeros((self.n+2,2)), np.zeros((self.n+2,3)) # we allow two more evaluations and crop at the end because each loop step can trigger up to 3 evaluations
        self.points[:2] = np.array([[self.traj.bounds[0,0],self.traj.bounds[1,0]], [self.traj.bounds[1,1],self.traj.bounds[1,1]]],dtype=float)
        self.res[0,1:], self.res[1,1:] = self.traj.evaluate(self.points[0]-self.traj.p0), self.traj.evaluate(self.points[1]-self.traj.p0)
        n_call = 2
        self.L = [(0,1,max(self.res[0,-1],self.res[1,-1]))] # list of segments to explore (extreme1 index, extreme2 index, score = max of scores at both extreme points)
        L_rejected = [] # list of rejected segments, if at some point self.L is empty, L_rejected is explored
        while n_call<self.n and len(self.L)>0:
            s = np.zeros((5),dtype=int)
            s[1], s[2], _ = self.L.pop()
            # s[1] and s[2] are the indices of the parents, s[0] the index of the middle, and s[3] and s[4] the other two vertices of the rectangle
            lvl = 1 + max(self.res[s[1],0],self.res[s[2],0]) # taking the max is overkill, both have the same value in fractal_grid but it conveys the idea
            mid, point1, point2 = (self.points[s[1]]+self.points[s[2]])/2, np.array([self.points[s[1],0],self.points[s[2],1]]), np.array([self.points[s[2],0],self.points[s[1],1]])
            #point1, point2 = mid + np.array([[0,-1],[1,0]]).dot(self.points[s[0]]-mid), mid - np.array([[0,-1],[1,0]]).dot(self.points[s[0]]-mid) #true rotation
            s[0] = n_call # the middle has never been previously evaluated
            self.points[s[0]] = mid
            self.res[s[0]] = np.concatenate([[lvl],self.traj.evaluate(mid-self.traj.p0)])
            n_call+=1
            # check whether point1 has already been evaluated, if not new evaluation (same for point2, mid has never already been visited for fractal_grid)
            if np.where(np.all(self.points[:n_call] == point1,axis=1))[0].size == 0:
                s[3] = n_call
                self.points[s[3]] = point1
                self.res[s[3]] = np.concatenate([[lvl], self.traj.evaluate(point1-self.traj.p0)])
                n_call+=1
            else :
                s[3] = np.where(np.all(self.points[:n_call] == point1,axis=1))[0][0]
            if np.where(np.all(self.points == point2,axis=1))[0].size == 0:
                s[4] = n_call
                self.points[s[4]] = point2
                self.res[s[4]] = np.concatenate([[lvl], self.traj.evaluate(point2-self.traj.p0)])
                n_call+=1
            else :
                s[4] = np.where(np.all(self.points == point2,axis=1))[0][0]
            # list of new candidates: accepted if both extremities have a different destination, and rejected otherwise, stop if lvl==lvlmax-1 then the children will bef lvlmax
            L_new = [(s[0],s[i],max(self.res[s[0],-1],self.res[s[i],-1])) for i in range(1,5) if (self.res[s[0],1] != self.res[s[i], 1] or lvl<lvl0) and lvl<lvlmax]
            L_newrejected = [(s[0],s[i],max(self.res[s[0],-1],self.res[s[i],-1])) for i in range(1,5) if (self.res[s[0],1] == self.res[s[i], 1] and lvl>=lvl0) and lvl<lvlmax]
            L_new.sort(key=lambda x:x[-1]); L_newrejected.sort(key=lambda x:x[-1]) # order the 1-3 child(ren) by increasing score so the biggest is on top
            self.L+=L_new; L_rejected = L_newrejected + L_rejected # do I prefer to visit first the highest or the lowest level (here the second option)
            if len(self.L)==0 : # if no frontier segment remains to explore we can go deeper on non-frontier ones
                self.L += L_rejected; L_rejected =[] # Je ne perdrai pas une semaine à cause d'une deepcopy. Je ne perdrai pas une semaine à cause d'une deepcopy. Je ne perdrai pas ...
            if lvl == lvl0: # when getting at lvl0, order by -lvl and then by score, otherwise low levels end up at the bottom of candidates
                self.L.sort(key=lambda x:(-max(self.res[x[0],0],self.res[x[1],0]),x[-1])) # explore lower levels first, then higher score first
            elif greed == True: # actually just need to put the max on top, not sort the whole list at each addition O(N) vs O(NlogN)
                self.L.sort(key=lambda x:(x[-1])) # greedy search, for width search include level, otherwise you dive depth-first, faster with bisect.insort or even better heapq.merge
        self.points, self.res = self.points[:self.n], self.res[:self.n] # cutting the possible 2 last evaluations to get fair competition among methods
        if n_call < self.n: # if lvlmax is too small compared to self.n, might get 0s which break the logcolorbar
            self.points[n_call:], self.res[n_call:] = np.tile(self.points[0],self.n-n_call).reshape(self.n-n_call,2), np.tile(self.res[0],self.n-n_call).reshape(self.n-n_call,self.res.shape[1])
        return self.points.copy(), self.res.copy()


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