import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

class Planets :
    def __init__(self, N, radius, G):
        self.N = N # number of planets
        self.radius = radius #radius of planets
        self.G = G # gravity constant (multiplied by planet mass)
        self.loc = np.zeros((N,2)) # planets locations

    def gen_planets(self,bounds,centralisation=0,trymax=10):
        ntry = 0
        span = np.array([bounds[0,1]-bounds[0,0]-3*self.radius, bounds[1,1]-bounds[1,0]-3*self.radius]) # planets cannot be closer than 1.5*r to the border
        self.loc = np.zeros((self.N,2))
        while np.min(4*self.radius*np.eye(self.N)+np.linalg.norm(self.loc[None,:,:]-self.loc[:,None,:], axis = 2))<=3*self.radius and ntry<trymax :
            self.loc = (1-centralisation) * span * (np.random.random(size=(self.N,2))-0.5) + np.mean(bounds,axis=1)
            ntry+=1
        if np.min(4*self.radius*np.eye(self.N)+np.linalg.norm(self.loc[None,:,:]-self.loc[:,None,:], axis = 2)) <= 3*self.radius :
            print("Failed to generate {} NON-colliding planets in {} tries, should reduce their number, radius or enlarge the frame bounds".format(self.N, trymax))
        return self.loc        

    def A(self, pos):
        a = np.zeros(2)
        for i in range(self.N):
            d_i = self.loc[i] - pos
            a += d_i/np.linalg.norm(d_i, axis=-1)**3
        return self.G*a

    def potential(self, pos):
        pot = 0
        for i in range(self.N):
            d_i = self.loc[i]-pos[...,:]
            pot -= self.G/np.linalg.norm(d_i, axis=-1)
        return pot

class Trajectory(Planets) :
    def __init__(self, N, radius, G, bounds, dt, Tmax):
        Planets.__init__(self, N, radius, G)
        self.p0 = np.zeros(2) # initial position
        self.p = np.zeros(2) # shuttle current position
        self.v = np.zeros(2) # shuttle current speed
        self.bounds = bounds # boundaries of the trajectory frame
        self.dt = dt # sampling time
        self.Tmax = Tmax # maximum trajectory simulation time, if reached trajectory is considered infinite
        self.p_hist = [] # successive positions of the trajectory
        self.v_hist = [] # successive speeds

    def gen_shuttle(self, centralisation=0, trymax=10):
        self.p0 = self.loc[0,:]
        span = np.array([self.bounds[0,1]-self.bounds[0,0], self.bounds[1,1]-self.bounds[1,0]])
        ntry=0
        while np.min(np.linalg.norm(self.loc-self.p0[None,:], axis = 1))<=2*self.radius and ntry<trymax :
            self.p0 = (1-centralisation)*span*(np.random.random(size=(2))-0.5)+np.mean(self.bounds,axis=1)
            ntry+=1
        return self.p0
        
    def leapfrog(self):
        #https://en.wikipedia.org/wiki/Leapfrog_integration
        v12 = self.v + 0.5 * self.dt * self.A(self.p)
        pos1 = self.p + self.dt * v12
        v1 = v12 + 0.5 * self.dt * self.A(pos1)
        return pos1, v1

    def stop_cond(self,border=True):
        for i in range(self.N):
            if np.linalg.norm(self.p-self.loc[i,:])<self.radius:
                return i+1 # crashed in planet i
        if border and (self.p[0]<self.bounds[0,0] or self.p[0]>self.bounds[0,1] or self.p[1]<self.bounds[1,0] or self.p[1]>self.bounds[1,1]):
            return self.N + 1 # crashed on border
        if len(self.p_hist) * self.dt >= self.Tmax:
            return self.N + 2 # reached maximum time limit
        
    def compute_traj(self, border=True):
        while not self.stop_cond(border):
            self.p_hist.append(self.p)
            self.v_hist.append(self.v)
            self.p, self.v = self.leapfrog()
        self.p_hist.append(self.p)
        self.v_hist.append(self.v)
        return np.array(self.p_hist), np.array(self.v_hist)

    def reset(self,p0=None,v0=np.zeros(2),loc=None):
        if not p0 is None :
            self.p0 = p0
        if not loc is None :
            self.loc = loc
            self.N = self.loc.shape[0]
        self.p_hist, self.v_hist = [], []
        self.p, self.v = self.p0, v0

class GridSolve :
    def __init__(self,n,bounds,traj):
        self.n = n #grid resolution
        self.bounds = bounds # boundaries of solution exploration
        self.traj = traj # instance of Trajectory
        
        self.x, self.y = np.linspace(self.bounds[0,0], self.bounds[0,1], self.n), np.linspace(self.bounds[1,0], self.bounds[1, 1], self.n)
        self.xx, self.yy = np.meshgrid(self.x,self.y)
        self.xy = np.stack([np.repeat(self.x,self.y.size).reshape((self.x.size,self.y.size)),np.tile(self.y,self.x.size).reshape((self.x.size,self.y.size))],axis=-1)
        
        self.pot = self.traj.potential(self.xy)
        self.cond = np.zeros((self.x.size,self.y.size))
        self. score = np.zeros((self.x.size,self.y.size))

        self.opt_ind = np.zeros((2,1))
        self.opt = np.zeros(2)
        self.vbest = np.zeros(2)
        self.Pbest = np.zeros((2,1))

    def compute_grid(self, border=True):
        for i in range(self.x.size):
            for j in range(self.y.size):
                self.traj.reset(v0=np.array([self.x[i],self.y[j]])-self.traj.p0)
                P, _ = self.traj.compute_traj(border)
                self.cond[i,j] = self.traj.stop_cond(border)
                self.score[i,j] = (P.shape[0]-1)*self.traj.dt

    def vbest_grid(self):
        self.opt_indices = np.array(np.where(self.score == self.score.max())).T
        self.opt = self.opt_indices[np.random.randint(self.opt_indices.shape[0])]
        self.vbest = self.xy[self.opt[0],self.opt[1]] - self.traj.p0
        print('{} maximum(s) - proposal : v = [{:.2f},{:.2f}]'.format(self.opt_indices.shape[0],self.vbest[0],self.vbest[1]))
        self.traj.reset(v0=self.vbest)
        self.Pbest, _ = self.traj.compute_traj()

class Display(GridSolve) :
    def __init__(self, n, traj, bounds = None, ntraj = None):
        if bounds is None :
            bounds = traj.bounds # useful to zoom in on a region of interest for grid search while the crash boundaries remain untouched
        if ntraj is None:
            ntraj = n # useful to have different values in the case of different boundaries for the trajectory computation and grid explorations

        GridSolve.__init__(self, n, bounds, traj)

        self.ntraj = ntraj
        self.trajx, self.trajy = np.linspace(self.traj.bounds[0,0], self.traj.bounds[0,1], self.ntraj), np.linspace(self.traj.bounds[1,0], self.traj.bounds[1, 1], self.ntraj)
        self.trajxx, self.trajyy = np.meshgrid(self.trajx,self.trajy)
        self.trajxy = np.stack([np.repeat(self.trajx,self.trajy.size).reshape((self.trajx.size,self.trajy.size)),np.tile(self.trajy,self.trajx.size).reshape((self.trajx.size,self.trajy.size))],axis=-1)

        self.pot = self.traj.potential(self.trajxy)

        self.pot_min = -self.traj.N*self.traj.G/self.traj.radius # minimum potential for colorbars, approximately potential value at planet surface
        self.colormap = colors.ListedColormap(['orchid', 'mediumorchid', 'darkorchid', 'blueviolet', 'rebeccapurple', 'purple', 'darkmagenta','mediumvioletred','hotpink'][:self.traj.N]+['navy', 'crimson'])
        self.colorbar_ticklabels = ['planet {}'.format(i+1) for i in range(self.traj.N)] + ['border', '(infinite)']
    
    def plot_config(self, figsize=(4,4)):
        fig, ax = plt.subplots(num=0,figsize=figsize,subplot_kw = {'aspect':1})
        c = ax.pcolormesh(self.trajxx, self.trajyy, self.pot.T, cmap='binary',vmin=self.pot_min,vmax=0)
        ax.axis([self.traj.bounds[0,0], self.traj.bounds[0,1], self.traj.bounds[1,0], self.traj.bounds[1,1]])
        ax.set_xticks([]),ax.set_yticks([])
        for i in range(self.traj.N):
            ax.add_patch(plt.Circle((self.traj.loc[i,0] , self.traj.loc[i,1]),self.traj.radius,color='k'))
        ax.scatter(self.traj.p0[0],self.traj.p0[1],c='r',marker='*',s=100)
        plt.show()

    def plot4(self, figsize = (16,16), save = False):
        fig, axs = plt.subplots(2, 2, figsize=figsize,layout='constrained',subplot_kw = {'aspect':1})
        
        axs[0,0].set_title('Initial configuration : p0 = [{}, {}]'.format(self.traj.p0[0],self.traj.p0[1]))
        axs[0,0].pcolormesh(self.trajxx, self.trajyy, self.pot.T, cmap='binary',vmin=self.pot_min,vmax=0)
        axs[0,0].axis([self.traj.bounds[0,0], self.traj.bounds[0,1], self.traj.bounds[1,0], self.traj.bounds[1,1]])
        axs[0,0].set_xticks([]); axs[0,0].set_yticks([])
        for i in range(self.traj.N):
            axs[0,0].add_patch(plt.Circle((self.traj.loc[i,0] , self.traj.loc[i,1]),self.traj.radius,color='k'))
        axs[0,0].scatter(self.traj.p0[0],self.traj.p0[1],c='r',marker='*',s=100)
        axs[0,0].add_patch(plt.Rectangle((self.bounds[0,0],self.bounds[1,0]),width=self.bounds[0,1]-self.bounds[0,0],height=self.bounds[1,1]-self.bounds[1,0],color='w',fill=False))
        
        axs[1,0].set_title('Trajectory proposition [{}, {}] : v = [{:.2f}, {:.2f}] : T = {:.2f}'.format(self.opt[0],self.opt[1],self.vbest[0],self.vbest[1],self.score[self.opt[0],self.opt[1]]))
        axs[1,0].pcolormesh(self.trajxx, self.trajyy, self.pot.T, cmap='binary',vmin=self.pot_min,vmax=0,alpha=0.8)
        axs[1,0].axis([self.traj.bounds[0,0], self.traj.bounds[0,1], self.traj.bounds[1,0], self.traj.bounds[1,1]])
        axs[1,0].set_xticks([]),axs[1,0].set_yticks([])
        for i in range(self.traj.N):
            axs[1,0].add_patch(plt.Circle((self.traj.loc[i,0] , self.traj.loc[i,1]),self.traj.radius,color='k'))
        axs[1,0].scatter(self.traj.p0[0],self.traj.p0[1],c='r',marker='x',s=40,zorder=3)
        axs[1,0].arrow(self.traj.p0[0],self.traj.p0[1],self.vbest[0],self.vbest[1],color='m',head_width=0.05,alpha=0.5,length_includes_head=True,zorder=2)
        axs[1,0].plot(self.Pbest[:,0],self.Pbest[:,1],linewidth=1,zorder=1)
        
        axs[0,1].set_title('Final destination')
        ms = axs[0,1].pcolormesh(self.xx, self.yy, self.cond.T, cmap = self.colormap, vmin=0.5, vmax=self.traj.N+2.5)
        axs[0,1].axis([self.xx.min(), self.xx.max(), self.yy.min(), self.yy.max()])
        axs[0,1].set_xticks([]),axs[0,1].set_yticks([])
        cbar = fig.colorbar(ms,ticks = np.arange(self.traj.N+2)+1)
        cbar.ax.set_yticklabels(self.colorbar_ticklabels,rotation='vertical',verticalalignment='center')
        for i in range(self.traj.N):
            axs[0,1].add_patch(plt.Circle((self.traj.loc[i,0], self.traj.loc[i,1]), self.traj.radius,color='k',alpha=0.2))
            axs[0,1].text(self.traj.loc[i,0] , self.traj.loc[i,1],'{}'.format(i+1),horizontalalignment='center',verticalalignment='center',c='w',alpha=0.5,size=20)
        axs[0,1].scatter((self.traj.p0+self.vbest)[0],(self.traj.p0+self.vbest)[1], s=80, facecolors='none', edgecolors='w',linewidths=2,alpha=0.5)
        axs[0,1].scatter(self.traj.p0[0],self.traj.p0[1],c='w',marker='+',s=100)
        
        axs[1,1].set_title('Travel time');
        pcm2 = axs[1,1].pcolormesh(self.xx, self.yy, self.score.T,cmap='managua', vmin=1, norm='log')
        axs[1,1].scatter((self.traj.p0+self.vbest)[0],(self.traj.p0+self.vbest)[1], s=80, facecolors='none', edgecolors='w',linewidths=2,alpha=0.5)
        axs[1,1].axis([self.xx.min(), self.xx.max(), self.yy.min(), self.yy.max()])
        axs[1,1].set_xticks([]),axs[1,1].set_yticks([])
        fig.colorbar(pcm2, ax=axs[1,1])

        if save :
            np.save('planets_loc/[{}, {}].npy'.format(self.traj.p0[0],self.traj.p0[1]),self.traj.loc)
            fig.savefig('figs/[{}, {}] - {} ({}).png'.format(self.traj.p0[0],self.traj.p0[1],self.traj.N,self.n),bbox_inches='tight')