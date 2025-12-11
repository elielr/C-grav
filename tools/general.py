import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

plt.style.use('dark_background')    

class Planets :
    def __init__(self, N, bounds, radius, G):
        self.N = N # number of planets
        self.bounds = bounds # boundaries of the planet generation
        self.span = np.array([self.bounds[0,1]-self.bounds[0,0], self.bounds[1,1]-self.bounds[1,0]])
        self.radius = radius #radius of planets
        self.G = G # gravity constant (multiplied by planet mass)
        self.loc = np.zeros((N,2)) # planets locations

    def gen_planets(self,bounds,centralisation=0,trymax=10):
        ntry = 0
        self.loc = np.zeros((self.N,2))
        while np.min(4*self.radius*np.eye(self.N)+np.linalg.norm(self.loc[None,:,:]-self.loc[:,None,:], axis = 2))<=3*self.radius and ntry<trymax :
            self.loc = (1-centralisation) * self.span * (np.random.random(size=(self.N,2))-0.5) + np.mean(bounds,axis=1)
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
        Planets.__init__(self, N, bounds, radius, G)
        self.p0 = np.zeros(2) # initial position
        self.p = np.zeros(2) # shuttle current position
        self.v = np.zeros(2) # shuttle current speed
        self.dt = dt # sampling time
        self.Tmax = Tmax # maximum trajectory simulation time, if reached trajectory is considered infinite
        self.p_hist = [] # successive positions of the trajectory
        self.v_hist = [] # successive speeds

    def gen_shuttle(self, centralisation=0, trymax=10):
        self.p0 = self.loc[0,:]
        ntry=0
        while np.min(np.linalg.norm(self.loc-self.p0[None,:], axis = 1))<=2*self.radius and ntry<trymax :
            self.p0 = (1-centralisation)*self.span*(np.random.random(size=(2))-0.5)+np.mean(self.bounds,axis=1)
            ntry+=1
        return self.p0
        
    def leapfrog(self):
        #https://en.wikipedia.org/wiki/Leapfrog_integration
        v12 = self.v + 0.5 * self.dt * self.A(self.p)
        pos1 = self.p + self.dt * v12
        v1 = v12 + 0.5 * self.dt * self.A(pos1)
        return pos1, v1

    def stop_cond(self, border=True):
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

    def evaluate(self,v0,border=True):
        self.reset(v0=v0)
        P, _ = self.compute_traj(border)
        return np.array([self.stop_cond(border), (P.shape[0]-1)*self.dt])       

class Display:
    def __init__(self, traj, n=1000, color_list=['orchid', 'mediumorchid', 'darkorchid', 'blueviolet', 'rebeccapurple', 'purple', 'darkmagenta','mediumvioletred','hotpink']):
        self.n = n
        self.traj = traj

        self.x, self.y = np.linspace(self.traj.bounds[0,0], self.traj.bounds[0,1], self.n), np.linspace(self.traj.bounds[1,0], self.traj.bounds[1, 1], self.n)
        self.xx, self.yy = np.meshgrid(self.x,self.y)
        self.xy = np.stack([np.repeat(self.x,self.y.size).reshape((self.x.size,self.y.size)),np.tile(self.y,self.x.size).reshape((self.x.size,self.y.size))], axis=-1)
        
        self.pot = self.traj.potential(self.xy)
        self.pot_min = -self.traj.N*self.traj.G/self.traj.radius # minimum potential for colorbars, approximately potential value at planet surface

        self.color_list = color_list[:self.traj.N]+["#081838", "#8FE8FF"]
        self.colormap = colors.ListedColormap(self.color_list)
        self.colorbar_ticklabels = ['planet {}'.format(i+1) for i in range(self.traj.N)] + ['border', '(infinite)']

    def plot_config(self,title=False,ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(4,4),layout='constrained',subplot_kw = {'aspect':1})
        if title:
            ax.set_title('Initial configuration : p0 = [{}, {}]'.format(self.traj.p0[0],self.traj.p0[1]))
        ax.pcolormesh(self.xx, self.yy, self.pot.T, cmap='binary',vmin=self.pot_min,vmax=0)
        ax.axis([self.traj.bounds[0,0], self.traj.bounds[0,1], self.traj.bounds[1,0], self.traj.bounds[1,1]])
        ax.set_xticks([]),ax.set_yticks([])
        for i in range(self.traj.N):
            ax.add_patch(plt.Circle((self.traj.loc[i,0] , self.traj.loc[i,1]),self.traj.radius,color='k'))
        ax.scatter(self.traj.p0[0],self.traj.p0[1],c='gold',marker='*',s=5e-4*min(ax.get_window_extent().width, ax.get_window_extent().height)**2)

    def plot_traj(self,v0,ax,border=True):
        self.traj.reset(v0=v0)
        P,_ = self.traj.compute_traj(border)
        ax.set_title('v = [{:.2f}, {:.2f}] : T = {:.2f}'.format(v0[0],v0[1],(P.shape[0]-1)*self.traj.dt))
        ax.pcolormesh(self.xx, self.yy, self.pot.T, cmap='binary',vmin=self.pot_min,vmax=0)
        ax.axis([self.traj.bounds[0,0], self.traj.bounds[0,1], self.traj.bounds[1,0], self.traj.bounds[1,1]])
        ax.set_aspect((self.y.max()-self.y.min())/(self.x.max()-self.x.min()))
        ax.set_xticks([]),ax.set_yticks([])
        for i in range(self.traj.N):
            ax.add_patch(plt.Circle((self.traj.loc[i,0] , self.traj.loc[i,1]),self.traj.radius,color='k'))
        ax.scatter(self.traj.p0[0],self.traj.p0[1],c='gold',marker='x',s=40,zorder=3)
        ax.arrow(self.traj.p0[0],self.traj.p0[1],v0[0],v0[1],color='r',head_width=0.05,alpha=0.8,length_includes_head=True,zorder=2)
        ax.plot(P[:,0],P[:,1],linewidth=1,zorder=1)