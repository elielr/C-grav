import numpy as np
import matplotlib.pyplot as plt

from tools.general import Display

class LineSearch:
    """ Selecting a random direction* and either make a uniform search or zoom in on a frontier where the score function is unbounded**.

    *Taking initial speeds in zones of different destination so at least one frontier exists between them (likely an infinity, see below)
    **So far just an intuition, between a initial speed whose trajectory ends in planet A and another that ends in planet B (or the border)
    there should be an infinity of discontinuities in both destination and score on the segment in which arbitrary high score are achieved.
    However these fractal strata can get very thin very fast and therefore hard to find if not impossible with double precision.
    """
    def __init__(self, traj, n=100, v1=None, v2=None):
        self.n = n # total number of evaluations omitting those at v1 and v2
        self.traj = traj
        self.eval = np.zeros((n+2,3),dtype='float') # stores the linear abscisse, the destination and the score associated
        self.V = np.zeros((n+2,2))

        self.v0 = np.zeros((2,2))
        if not v1 is None :
            self.v0[0] = v1
        self.eval[0, 1:] = self.ending(self.v0[0])
        self.eval[-1,0] = 1
        if not v2 is None:
            self.v0[1] = v2
            self.eval[-1,1:] = self.ending(self.v0[1])
        else :
            self.eval[-1,1] = self.eval[0, 1]
            while self.eval[-1,1] == self.eval[0,1]: # we want v0 and v1 to have different destinations to look at the frontiers on the line
                self.v0[1] = self.traj.span*(np.random.random(size=(2))-0.5)+np.mean(self.traj.bounds,axis=1) - self.traj.p0
                self.eval[-1,1:] = self.ending(self.v0[1])
        
        self.opt_ind = np.zeros((2,1))
        self.opt = np.zeros(2)
        self.vbest = np.zeros(2)

    def ending(self, v, border=True):
        self.traj.reset(v0=v)
        P, _ = self.traj.compute_traj(border)
        return np.array([self.traj.stop_cond(border), (P.shape[0]-1)*self.traj.dt])

    #v = lambda self, a : (1-a)*self.v0[0] + a*self.v0[1]
    v = lambda self, a : np.array([(1-a)*self.v0[0,0]+a*self.v0[1,0], (1-a)*self.v0[0,1]+a*self.v0[1,1]]).T # moins classe mais efficace

    evaluate = lambda self, a : np.concatenate([[a],self.ending(self.v(a))]) # np.concatenate([A[:,None],self.endings(self.v(A))]) if we get a parallel version of ending

    ### The Methods ###
    
    # https://en.wikipedia.org/wiki/Line_search doesn't apply here, even 0-order golden-section, our function is absolutely not unimodal, and random local maxima will be disappointing
    
    def uniform_search(self):
        self.eval[1:-1,0] = np.linspace(0,1,self.n+2)[1:-1]
        for i in range(1,self.n+1):
            self.eval[i,1:] = self.ending(self.v(self.eval[i,0]))
        self.V = self.v(self.eval[:,0])
        return self.eval.copy()
           
    def dicho_frontiersearch(self):
        # v1 and v2 are in zones of different destinations (i.e. trajectory crash region), we zoom in on a (any) frontier between two detinations
        a0, a1 = 0, self.n+1
        self.eval[1:-1] = np.zeros((self.n,3))
        for i in range(1,min(self.n+1,50)): # 0.5**50 = 1e-15 no need to go too close to double precision since position takes values close to 1
            self.eval[i] = self.evaluate((self.eval[a0,0]+self.eval[a1,0])/2)
            if self.eval[i,1] == self.eval[a0,1]:
                a0 = i
            else :
                a1 = i
        self.V = self.v(self.eval[:,0])
        return self.eval.copy()

    def mixed_search(self,n_step=100):
        # if n_step == 1 it is the dicho_frontiersearch, if n_step == self.n it is the uniform_search, inbetween it's a balance
        n_zoom = min(self.n//n_step, -int(np.emath.logn(n_step+1, 0.5**50))) # no need to zoom ad eternam due to float double precision
        self.eval[1:-1] = np.zeros((self.n,3))      
        a0, a1 = 0, self.n+1
        for i in range(n_zoom):
            self.eval[i*n_step+1:(i+1)*n_step+1,0] = np.linspace(self.eval[a0,0],self.eval[a1,0],n_step+2)[1:-1]
            for j in range(1,n_step+1):
                self.eval[i*n_step+j,1:] = self.ending(self.v(self.eval[i*n_step+j,0]))
            # mask keeps track at zoom level i of which points have a different destination than their right neighbor and the score of these transfrontier segments (max score at its extrmities)
            mask = np.zeros(self.n+2)
            mask[i*n_step+1:(i+1)*n_step] = self.eval[i*n_step+1:(i+1)*n_step,1] != self.eval[i*n_step+2:(i+1)*n_step+1,1]
            mask *= np.maximum(self.eval[:,2],np.roll(self.eval[:,2],-1))
            mask[a0] = (self.eval[a0,1] != self.eval[i*n_step+1,1]) * np.max([self.eval[a0,2], self.eval[i*n_step+1,2]])
            mask[(i+1)*n_step] = (self.eval[a1,1] != self.eval[(i+1)*n_step,1]) * np.max([self.eval[a1,2], self.eval[(i+1)*n_step,2]])
            # select zoommed in region of interest
            a2 = np.argmax(mask)
            if a2 == a0:
                a1 = i*n_step+1
            elif a2 == (i+1)*n_step:
                a0 = (i+1)*n_step
            else:
                a0, a1 = a2, a2+1
        self.V = self.v(self.eval[:,0])
        return self.eval.copy()
    
    def dicho_multisearch(self):
        # since we only go up to 0.5**50 we can zoom in on multiple frontiers if the three crashes are different a0!=a2!=a1
        self.eval[1:-1] = np.zeros((self.n,3))
        L = [(0,self.n+1)]
        n_call = 0
        while n_call<self.n and len(L)>0:
            a0, a1 = L.pop()
            if (self.eval[a1,0]-self.eval[a0,0])>0.5**50:
                n_call+=1
                mid = (self.eval[a0,0]+self.eval[a1,0])/2
                self.eval[n_call,:] = self.evaluate(mid)
                if self.eval[n_call,1] != self.eval[a1,1]:
                    L.append((n_call,a1))
                if self.eval[n_call,1] != self.eval[a0,1]:
                    L.append((a0,n_call)) # left first to match dicho_frontiersearch
        self.V = self.v(self.eval[:,0])
        return self.eval.copy()

    def mixed_mutlisearch(self,n_step=100):
        self.eval[1:-1] = np.zeros((self.n,3))
        L = [(0,self.n+1,0.)]
        n_call = 0
        while n_call<self.n and len(L)>0:
            a0, a1, _ = L.pop() # L.pop(0) for a width first search but we want to explore around one maximum, not get a general idea of the evolution 
            if (self.eval[a1,0]-self.eval[a0,0])>0.5**50: # we don't want to get too close to double precision
                self.eval[n_call+1:n_call+n_step+1,0] = np.linspace(self.eval[a0,0],self.eval[a1,0],n_step+2)[1:-1]
                # left border case
                self.eval[n_call+1] = self.evaluate(self.eval[n_call+1,0])
                L_new = [(a0,n_call+1,max(self.eval[a0,2],self.eval[n_call+1,2]))] if self.eval[a0,1]!=self.eval[n_call+1,1] else []
                # inside loop, compare to left neighbor
                for j in range(2,n_step+1):
                    self.eval[n_call+j] = self.evaluate(self.eval[n_call+j,0])
                    if self.eval[n_call+j,1]!=self.eval[n_call+j-1,1]:
                        L_new.append((n_call+j-1,n_call+j,max(self.eval[n_call+j-1,2],self.eval[n_call+j,2])))
                # right border case
                if self.eval[n_call+n_step,1]!=self.eval[a1,1]:
                    L_new.append((n_call+n_step,a1,max(self.eval[n_call+n_step,2],self.eval[a1,2])))
                L_new.sort(key=lambda x:(x[-1],-x[0])) # sorting the new candidates, the best scores will be on top, in case of tie take the segment of smallest abscisse (so it matches mixed_search)
                L += L_new # putting new candidates list on top of full list, no full sorting we explore around one maximum
                n_call+=n_step
        self.V = self.v(self.eval[:,0])
        return self.eval.copy()

    ### Optimum ###
    
    def vbest_line(self, verbose = True):
        self.opt_indices = np.array(np.where(self.eval[:,2] == self.eval[:,2].max())[0])
        self.opt = self.opt_indices[np.random.randint(self.opt_indices.shape[0])]
        self.vbest = self.v(self.eval[self.opt,0])
        if verbose:
            print('{} maximum(s) - proposal [{}] : v = [{:.2f},{:.2f}]'.format(self.opt_indices.shape[0],self.opt,self.vbest[0],self.vbest[1]))
        self.traj.reset(v0=self.vbest)



class DisplayLineSearch(Display, LineSearch):
    def __init__(self, traj, n=100, v1=None, v2=None):
        Display.__init__(self, traj)
        LineSearch.__init__(self, traj, n, v1, v2)

    def plot_ls_config(self,title=False,ax=None):
        if ax is None :
            fig, ax = plt.subplots(figsize=(4,4),layout='constrained',subplot_kw = {'aspect':1})
        self.plot_config(title,ax)
        ax.plot(self.traj.p0[0]+self.v0[:,0],self.traj.p0[1]+self.v0[:,1],c='orange',linewidth=1)
    
    def plot1D(self,ev=None,ymin=0,ymax=0, ax=None):
        self.eval = self.eval if ev is None else ev
        ymax = np.max(self.eval[:,2][self.eval[:,2]<self.traj.Tmax]) if ymax==0 else ymax; ymin = np.min(self.eval[:,2][self.eval[:,2]>0]) if ymin==0 else ymin; 
        ymax = np.min([self.traj.Tmax, 10**int(np.floor(np.log10(ymax))) * (1+int(np.floor(ymax/10**int(np.floor(np.log10(ymax))))))])
        if ax is None:
            fig, ax = plt.subplots(figsize=(16,3),layout='constrained')
        ax.vlines(self.eval[:,0][self.eval[:,1] == self.traj.N+2], 0, ymax, colors="#8FE8FF",zorder=1)
        ax.vlines(self.eval[np.argmax(self.eval[:,2]),0], 0, self.eval[np.argmax(self.eval[:,2]),2], colors='r', zorder=2)
        ms = ax.scatter(self.eval[:,0][(self.eval[:,2]>0) * (self.eval[:,1] != self.traj.N+2)],self.eval[:,2][(self.eval[:,2]>0) * (self.eval[:,1] != self.traj.N+2)],
                        c=self.eval[:,1][(self.eval[:,2]>0) * (self.eval[:,1] != self.traj.N+2)],cmap=self.colormap,s=2,vmin=0.5,vmax=self.traj.N+2.5,zorder=3)
        if self.eval[np.argmax(self.eval[:,2]),1] != self.traj.N+2:
            ax.scatter(self.eval[np.argmax(self.eval[:,2]),0],self.eval[np.argmax(self.eval[:,2]),2],s=80, facecolors='none', edgecolors='r',linewidths=2,zorder=2)
        ax.set_yscale('log')
        ax.set_ylim(ymin,ymax)
        ax.set_yticks(ticks=[10**i for i in range(int(np.floor(np.log10(np.max(self.eval[:,2]))+1)))]); ax.yaxis.set_minor_formatter(plt.NullFormatter())
        return ms
    
    def plot1D_plustraj(self,ev=None,ymin=0,ymax=0,axs=None):
        self.eval = self.eval if ev is None else ev
        self.vbest_line(verbose = False)
        if axs is None:
            fig, axs = plt.subplots(1,2,figsize=(16,3),layout='constrained',width_ratios=[4, 1])
        ms = self.plot1D(self.eval,ymin,ymax,axs[0])
        self.plot_traj(self.vbest,axs[1])
        axs[1].plot(self.traj.p0[0]+self.v0[:,0],self.traj.p0[1]+self.v0[:,1],c='orange',linewidth=1,alpha=0.2,zorder=1.5)
        return ms
    
    def plot_list(self,evals,titles=None,save=False):
        n_eval = evals.shape[0]
        fig, axs = plt.subplots(n_eval,2,figsize=(16,3*n_eval),layout='constrained', width_ratios=[4, 1])
        fig.suptitle('p0 = [{}, {}] \n v0 = [{}, {}] - v1 = [{}, {}]\n'.format(self.traj.p0[0],self.traj.p0[1],self.v0[0,0],self.v0[0,1],self.v0[1,0],self.v0[1,1]))
        ymin, ymax = np.min(evals[:,:,2][(evals[:,:,2]>0) * (evals[:,:,1] != self.traj.N+2)]), np.max(evals[:,:,2][(evals[:,:,2]>0) * (evals[:,:,1] != self.traj.N+2)])        
        for i in range(n_eval):
            ms = self.plot1D_plustraj(evals[i],ymin,ymax,axs=axs[i,:])
            title_i = '{} maximum(s) - proposal [{}]'.format(self.opt_indices.shape[0],self.opt)
            if not titles is None and i<len(titles):
                title_i = titles[i] + ' - ' + title_i
            axs[i,0].set_title(title_i)
        cbar = fig.colorbar(ms,ticks = np.arange(self.traj.N+2)+1,ax=axs[:,0].ravel().tolist(),aspect=50,pad=0)
        cbar.ax.set_yticklabels(self.colorbar_ticklabels,rotation='vertical',verticalalignment='center')
        if save :
            np.save('save/planets_loc/[{}, {}].npy'.format(self.traj.p0[0],self.traj.p0[1]),self.traj.loc)
            np.save('save/line/[{}, {}].npy'.format(self.traj.p0[0],self.traj.p0[1]),self.v0)
            fig.savefig('figs/[{}, {}] - 1D (l{}) .png'.format(self.traj.p0[0],self.traj.p0[1],self.n),bbox_inches='tight')  

    def scatter2D_crash(self,ax):
        ax.set_title('Final Destination')
        sc = ax.scatter(self.traj.p0[0]+self.V[:,0],self.traj.p0[1]+self.V[:,1],c=self.eval[:,1], cmap = self.colormap, vmin=0.5, vmax=self.traj.N+2.5)
        ax.scatter((self.traj.p0+self.vbest)[0],(self.traj.p0+self.vbest)[1], s=80, facecolors='none', edgecolors='r',linewidths=2)
        ax.axis([self.xx.min(), self.xx.max(), self.yy.min(), self.yy.max()])
        ax.set_xticks([]),ax.set_yticks([])
        return sc

    def scatter2D_score(self,ax):
        ax.set_title('Flight time')
        sc = ax.scatter(self.traj.p0[0]+self.V[:,0],self.traj.p0[1]+self.V[:,1],c=self.eval[:,2], cmap='managua', norm='log')
        ax.scatter((self.traj.p0+self.vbest)[0],(self.traj.p0+self.vbest)[1], s=80, facecolors='none', edgecolors='r',linewidths=2)
        ax.axis([self.xx.min(), self.xx.max(), self.yy.min(), self.yy.max()])
        ax.set_xticks([]),ax.set_yticks([])
        return sc

    def plot4(self, figsize = (16,16), save = False):
        fig, axs = plt.subplots(2, 2, figsize=figsize,layout='constrained',subplot_kw = {'aspect':1})
        self.plot_ls_config(title=True, ax=axs[0,0])
        self.plot_traj(self.vbest,ax=axs[1,0])
        sc1 = self.scatter2D_crash(axs[0,1])
        cbar1 = fig.colorbar(sc1,ticks = np.arange(self.traj.N+2)+1); cbar1.ax.set_yticklabels(self.colorbar_ticklabels,rotation='vertical',verticalalignment='center')
        sc2 = self.scatter2D_score(axs[1,1])
        cbar2 = fig.colorbar(sc2,extend='max',ax=axs[1,1]) if np.max(self.eval[:,2])==self.traj.Tmax else fig.colorbar(sc2,ax=axs[1,1])
        cbar2.set_ticks(ticks=[10**i for i in range(int(np.floor(np.log10(np.max(self.eval[:,2]))+1)))]); cbar2.ax.yaxis.set_minor_formatter(plt.NullFormatter())
        if save :
            np.save('save/planets_loc/[{}, {}].npy'.format(self.traj.p0[0],self.traj.p0[1]),self.traj.loc)
            np.save('save/line/[{}, {}].npy'.format(self.traj.p0[0],self.traj.p0[1]),self.v0)
            fig.savefig('figs/[{}, {}] - 2D (l{}).png'.format(self.traj.p0[0],self.traj.p0[1],self.traj.N,self.n),bbox_inches='tight')

# Future idea: multidir_search, reinitialize v0[1] multiple times to look for wider optimums in easier zones, should be it's own class, n/n_dir calls in ech direction