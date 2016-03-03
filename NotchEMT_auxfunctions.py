#-------------------------------------------------------------------------------------------------------------------#
#  Auxilary functions.
#  Writen by Marcelo Boareto: marceloboareto@gmail.com
#  Last update: 11/2015
#-------------------------------------------------------------------------------------------------------------------#
from PyDSTool import *
from PyDSTool.Toolbox import phaseplane as pp
from matplotlib import cm
import numpy as np
import random as rand
import matplotlib.pyplot as plt
import random as rand
import copy as cp
from scipy.ndimage import measurements
import matplotlib.colors as cl
import math

def parameters(onecell=False):
    dic =  {'km' : 5.0e-1
           ,'kP' : 1.0e+2
           #----------EMT circuit----------
           #-- see table SI4a (L>Y) -- 
           ,'gu200' : 2.1e+3, 'gZ': 1.1e+1, 'Z0u200': 2.2e+5, 'Z0Z': 2.5e+4  
           # -- see table SI4b --
           ,'nZu200': 3.0e+0, 'nZZ': 2.0e+0, 'nu200' : 6.0e+0, 'nSu200': 2.0e+0, 'nSZ': 2.0e+0  
           ,'lZu200': 1.0e-1, 'lZZ': 7.5e+0, 'lSu200': 1.0e-1, 'lSZ'   : 1.0e+1
           ,'ku200' : 5.0e-2, 'kZ' : 1.0e-1
           ,'S0u200': 1.8e+5, 'S0Z': 1.8e+5
           ,'u0200' : 1.0e+4
           # -- see table SI4c --
           ,'nSu34' : 1.0e+0, 'nSS': 1.0e+0, 'nu34' : 2.0e+0, 'nI' : 2.0e+0   
           ,'lSu34' : 1.0e-1, 'lSS': 1.0e-1, 'lZu34': 2.0e-1, 'lIS': 6.5e+0
           ,'ku34'  : 5.0e-2, 'kS' : 1.25e-1
           ,'gu34'  : 1.35e+3,'gS' : 9.0e+1
           ,'S0u34' : 3.0e+5, 'S0S': 2.0e+5
           ,'Z0u34' : 6.0e+5
           ,'u034'  : 1.0e+4
           ,'I0S'   : 3.0e+2
           #-------------------------------------------
           #----------Notch Signaling circuit----------
           ,'k'  : 1.0e-1, 'kI' : 5.0e-1                                    
           ,'kc' : 1.0e-4, 'kt' : 1.0e-5                                   
           ,'p'  : 2.0e+0, 'pf' : 1.0            
           ,'gN' : 0.8e+1, 'gD' : 7.0e+1, 'gJ' : 2.0e+1     
           ,'I0' : 1.0e+2                                   
           ,'Nt' : 0.0e+0, 'Dt' : 0.0e+0, 'Jt' : 0.0e+0 
           ,'Nn' : 0.0e+0, 'Dn' : 0.0e+0, 'Jn' : 0.0e+0 
           ,'ln' : 7.0e+0, 'ld' : 0.0e+0, 'lj' : 2.0e+0   
           ,'ldf': 3.0,    'ljf': 0.3
           ,'It' : 0.0
           }
    # ---------see table SI1---------
    if onecell:
         dic.update({'l0': 1.0e+0,'l1': 6.0e-1,'l2': 3.0e-1
		    ,'l3': 1.0e-1,'l4': 5.0e-2,'l5': 5.0e-2,'l6': 5.0e-2
		    ,'gm0': 0.0e+0,'gm1': 4.0e-2,'gm2': 2.0e-1
                    ,'gm3': 1.0e+0,'gm4': 1.0e+0,'gm5': 1.0e+0,'gm6': 1.0e+0
                    ,'gu0': 0.0e+0,'gu1': 1*5.0e-3,'gu2': 2*5.0e-2,'gu3': 3*5.0e-1
                    ,'gu4': 4*5.0e-1,'gu5': 5*5.0e-1,'gu6': 6*5.0e-1
                    })
      
    else:
         dic.update({'l' : [1.0e+0, 6.0e-1, 3.0e-1, 1.0e-1, 5.0e-2, 5.0e-2, 5.0e-2]
                    ,'gm': [0.0e+0, 4.0e-2, 2.0e-1, 1.0e+0, 1.0e+0, 1.0e+0, 1.0e+0]
                    ,'gu': [0.0e+0, 1*5.0e-3, 2*5.0e-2, 3*5.0e-1, 4*5.0e-1, 5*5.0e-1, 6*5.0e-1]
                    ,'HS' : HS, 'Pl' : Pl, 'Py' : Py
                    })
    return dic
  
def equations(onecell=False):
    if onecell:
          return {'W' : 'gu200*HS(Z,Z0u200,nZu200,lZu200)*HS(S,S0u200,nSu200,lSu200) - gZ*HS(Z,Z0Z,nZZ,lZZ)*HS(S,S0Z,nSZ,lSZ)*Py(W,km,6) - gJ*HS(I,I0,p,lj)*Py(W,km,5) - ku200*W'
                 ,'Y' : 'gu34*HS(S,S0u34,nSu34,lSu34)*HS(Z,Z0u34,nu34,lZu34) - gS*HS(S,S0S,nSS,lSS)*HS(I,I0S,nI,lIS)*HS(It,I0S,nI,lIS)*Py(Y,km,2) - gN*HS(I,I0,p,ln)*Py(Y,km,2) - gD*HS(I,I0,p,ld)*Py(Y,km,3) - ku34*Y'
                 ,'Z' : 'kP*gZ*HS(Z,Z0Z,nZZ,lZZ)*HS(S,S0Z,nSZ,lSZ)*Pl(W,km,6)                    - kZ*Z'
                 ,'S' : 'kP*gS*HS(S,S0S,nSS,lSS)*HS(I,I0S,nI ,lIS)*HS(It,I0S,nI,lIS)*Pl(Y,km,2)  - kS*S'   
                 ,'N' : 'kP*gN*HS(I,I0,p,ln)*Pl(Y,km,2) - N*( (kc*D + kt*Dt)*HS(I,I0,pf,ldf) + (kc*J + kt*Jt)*HS(I,I0,pf,ljf) ) - k*N'
                 ,'D' : 'kP*gD*HS(I,I0,p,ld)*Pl(Y,km,3) - D*(  kc*N*HS(I,I0,pf,ldf) + kt*Nt ) - k*D'
                 ,'J' : 'kP*gJ*HS(I,I0,p,lj)*Pl(W,km,5) - J*(  kc*N*HS(I,I0,pf,ljf) + kt*Nt ) - k*J'
                 ,'I' : 'kt*N*( Dt*HS(I,I0,pf,ldf) + Jt*HS(I,I0,pf,ljf) ) - kI*I'
                 }
    else:
          return {'W' : 'gu200*HS(Z,Z0u200,nZu200,lZu200)*HS(S,S0u200,nSu200,lSu200) - gZ*HS(Z,Z0Z,nZZ,lZZ)*HS(S,S0Z,nSZ,lSZ)*Py(W,6,km,u0200) - gJ*HS(I,I0,p,lj)*Py(W,5,km,u0200) - ku200*W'
                 ,'Y' : 'gu34*HS(S,S0u34,nSu34,lSu34)*HS(Z,Z0u34,nu34,lZu34) - gS*HS(S,S0S,nSS,lSS)*HS(I,I0S,nI,lIS)*HS(It,I0S,nI,lIS)*Py(Y,2,km,u034) - gN*HS(I,I0,p,ln)*Py(Y,2,km,u034) - gD*HS(I,I0,p,ld)*Py(Y,3,km,u034) - ku34*Y'
                 ,'Z' : 'kP*gZ*HS(Z,Z0Z,nZZ,lZZ)*HS(S,S0Z,nSZ,lSZ)*Pl(W,6,km,u0200) - kZ*Z'
                 ,'S' : 'kP*gS*HS(S,S0S,nSS,lSS)*HS(I,I0S,nI ,lIS)*HS(It,I0S,nI,lIS)*Pl(Y,2,km,u034)  - kS*S'   
                 ,'N' : 'kP*gN*HS(I,I0,p,ln)*Pl(Y,2,km,u034)  - N*( (kc*D + kt*(Dn + Dt))*HS(I,I0,pf,ldf) + (kc*J + kt*(Jn + Jt))*HS(I,I0,pf,ljf) ) - k*N'
                 ,'D' : 'kP*gD*HS(I,I0,p,ld)*Pl(Y,3,km,u034)  - D*(  kc*N*HS(I,I0,pf,ldf) + kt*(Nn + Nt) ) - k*D'
                 ,'J' : 'kP*gJ*HS(I,I0,p,lj)*Pl(W,5,km,u0200) - J*(  kc*N*HS(I,I0,pf,ljf) + kt*(Nn + Nt) ) - k*J'
                 ,'I' : 'kt*N*( (Dn + Dt)*HS(I,I0,pf,ldf) + (Jn + Jt)*HS(I,I0,pf,ljf) ) - kI*I'
                 }
    #return {'W' : 'gu200*HS(Z,Z0u200,nZu200,lZu200)*HS(S,S0u200,nSu200,lSu200) - gZ*HS(Z,Z0Z,nZZ,lZZ)*HS(S,S0Z,nSZ,lSZ)*Py(W,6,km,u0200,gu,gm) - gJ*HS(I,I0,p,lj)*Py(W,5,km,u0200,gu,gm) - ku200*W'
           #,'Y' : 'gu34*HS(S,S0u34,nSu34,lSu34)*HS(Z,Z0u34,nu34,lZu34) - gS*HS(S,S0S,nSS,lSS)*HS(I,I0S,nI,lIS)*HS(Iext,I0S,nI,lIS)*Py(Y,2,km,u034,gu,gm) - gN*HS(I,I0,p,ln)*Py(Y,2,km,u034,gu,gm) - gD*HS(I,I0,p,ld)*Py(Y,3,km,u034,gu,gm) - ku34*Y'
           #,'Z' : 'kP*gZ*HS(Z,Z0Z,nZZ,lZZ)*HS(S,S0Z,nSZ,lSZ)*Pl(W,6,km,u0200,l,gm) - kZ*Z'
           #,'S' : 'kP*gS*HS(S,S0S,nSS,lSS)*HS(I,I0S,nI ,lIS)*HS(Iext,I0S,nI,lIS)*Pl(Y,2,km,u034,l,gm)  - kS*S'   
           #,'N' : 'kP*gN*HS(I,I0,p,ln)*Pl(Y,2,km,u034, l,gm) - N*( (kc*D + kt*(Dn + Dt))*HS(I,I0,pf,ldf) + (kc*J + kt*(Jn + Jt))*HS(I,I0,pf,ljf) ) - k*N'
           #,'D' : 'kP*gD*HS(I,I0,p,ld)*Pl(Y,3,km,u034, l,gm) - D*(  kc*N*HS(I,I0,pf,ldf) + kt*(Nn + Nt) ) - k*D'
           #,'J' : 'kP*gJ*HS(I,I0,p,lj)*Pl(W,5,km,u0200,l,gm) - J*(  kc*N*HS(I,I0,pf,ljf) + kt*(Nn + Nt) ) - k*J'
           #,'I' : 'kt*N*( (Dn + Dt)*HS(I,I0,pf,ldf) + (Jn + Jt)*HS(I,I0,pf,ljf) ) - kI*I'
           #}
  
# Auxilary functions
def HS(X,X0,nX,lamb):
    return lamb + (1.0-lamb)/(1.0 + (X/X0)**nX)

def M(X,X0,i,n):
    return ((X/X0)**i)/((1. + (X/X0))**n)

def C(i,n):
    return gamma(n+1)/(gamma(n-i+1)*gamma(i+1))

def Py(X, n, k, u0, gu=[0.0e+0, 1*5.0e-3, 2*5.0e-2, 3*5.0e-1, 4*5.0e-1, 5*5.0e-1, 6*5.0e-1],
       gm=[0.0e+0, 4.0e-2, 2.0e-1, 1.0e+0, 1.0e+0, 1.0e+0, 1.0e+0]):
    v1 = 0
    v2 = 0
    for i in range(n):
        v1 += gu[i]*C(i,n)*M(X,u0,i,n)
        v2 += gm[i]*C(i,n)*M(X,u0,i,n)
    return v1/(v2+k)

def Pl(X, n, k, u0, l=[1.0e+0, 6.0e-1, 3.0e-1, 1.0e-1, 5.0e-2, 5.0e-2, 5.0e-2],
       gm=[0.0e+0, 4.0e-2, 2.0e-1, 1.0e+0, 1.0e+0, 1.0e+0, 1.0e+0]):
    v1 = 0
    v2 = 0
    for i in range(n):
        v1 +=  l[i]*C(i,n)*M(X,u0,i,n)
        v2 += gm[i]*C(i,n)*M(X,u0,i,n)
    return v1/(v2+k)


def vlim(c=None):
    if c=='E':
        return {'N': [0.0e+4, 3.0e+4], 'D': [0.0e+4, 0.2e+4], 'J': [0.0e+4, 1.0e+4], 'I': [0.0e+4, 0.002e+4]
               ,'W': [1.5e+4, 3.5e+4], 'Z': [0.0e+5, 0.1e+4], 'Y': [1.6e+4, 2.2e+4], 'S': [0.5e+5, 2.0e+5]
               }
    elif c=='E/M':
        return {'N': [3.0e+4, 4.0e+4], 'D': [0.0e+4, 0.1e+4], 'J': [0.1e+3, 1.0e+3], 'I': [0.5e+3, 1.5e+3]
               ,'W': [0.5e+4, 1.5e+4], 'Z': [1.0e+4, 1.5e+5], 'Y': [1.5e+4, 1.7e+4], 'S': [1.7e+5, 2.1e+5]
               }
    elif c=='M':
        return {'N': [2.0e+4, 3.0e+4], 'D': [0.0e+4, 0.1e+4], 'J': [2.0e+3, 5.0e+3], 'I': [0.4e+3, 1.6e+3]
               ,'W': [0.0e+4, 0.5e+4], 'Z': [4.0e+5, 8.5e+5], 'Y': [0.6e+4, 1.5e+4], 'S': [2.0e+5, 3.0e+5]
               }
    else:
        return {'N': [0.0e+4, 3.0e+4], 'D': [0.0e+4, 0.2e+4], 'J': [0.0e+4, 1.0e+4], 'I': [0.0e+4, 0.2e+4]
               ,'W': [0.0e+4, 4.0e+4], 'Z': [0.0e+5, 9.0e+5], 'Y': [0.6e+4, 2.4e+4], 'S': [5.0e+4, 3.0e+5]
               }


def functions():
    return {'HS': (['X','X0','nX','lamb'], 'lamb + (1.0-lamb)/(1.0 + (X/X0)**nX)')
           ,'M' : (['X','X0','i','n'], '((X/X0)**i)/((1. + (X/X0))**n)' )
           ,'C' : (['i','n'],'special_gamma(n+1)/(special_gamma(n-i+1)*special_gamma(i+1))'     )
           ,'Py': (['X','kd','n'],'sum(i, 0, 6, if([i]>n, 0, gu[i]*C([i],n)*M(X,u0200,[i],n)))/(sum(i, 0, 6, if([i]>n, 0, gm[i]*C([i],n)*M(X,u0200,[i],n))) + kd)')
           ,'Pl': (['X','kd','n'],'sum(i, 0, 6, if([i]>n, 0,  l[i]*C([i],n)*M(X,u0200,[i],n)))/(sum(i, 0, 6, if([i]>n, 0, gm[i]*C([i],n)*M(X,u0200,[i],n))) + kd)')
           }

#----------------------------------------------------------------------------------------#
# Euler method for solving EDO equations 
def euler_traj(eqs, p, pts=None, vlim=None, hexagonal=True, 
	       nsignal_dict={'N': ['D', 'J'], 'I': ['D', 'J'], 'D': ['N'], 'J': ['N']}):
    if pts==None:
        if vlim==None:
            print 'ERROR: Give me a starting point (pts) or the limits for a random start point (vlim)'
            return 0
        pts = {}   
        for j in eqs.keys():
            pts[j] = np.random.uniform(vlim[j][0],vlim[j][1],(p['n'],p['n'])) 

    pts_new = {}
    for t in range(int(p['t']/p['dt'])):
        for key in eqs.keys():
            if key in nsignal_dict.keys(): 
                for k in nsignal_dict[key]:
                    p[k+'n'] = nsignal_sum(p, pts, k, key, hexagonal=hexagonal)
            pts_new[key] = pts[key] + p['dt']*eval(eqs[key], p, pts)
        pts = pts_new
    return pts

#----------------------------------------------------------------------------------------#
# Sum the amount of proteins of the neighboring cells
def nsignal_sum(p, pts, k, key, hexagonal=True):
    n = p['n']
    fng_dic={'D': 'HS(I,I0,pf,ldf)'
            ,'J': 'HS(I,I0,pf,ljf)'
            }
    fng = 1.0
    if k=='N':
        fng = eval(fng_dic[key],p,pts)
    X = periodic_bcondition(pts[k]*fng, n)
    if hexagonal:
        return (1.0/6.0)*( X[0:n,1:(n+1)] + X[1:(n+1),0:n] + X[2:(n+2),1:(n+1)] + X[1:(n+1),2:(n+2)] 
                         + X[0:n,2:(n+2)] + X[2:(n+2),2:(n+2)] )
    return     (1.0/4.0)*( X[0:n,1:(n+1)] + X[1:(n+1),0:n] + X[2:(n+2),1:(n+1)] + X[1:(n+1),2:(n+2)]  )

#----------------------------------------------------------------------------------------#
# Expand the matrix from nxn to (n+2)x(n+2) where the extra rows and columns are chosen by 
# a periodic boundary condition
def periodic_bcondition(f, n):
    out = np.zeros((n+2,n+2))
    out[1:(n+1),1:(n+1)] = f
    out[0      ,1:(n+1)] = out[n      ,1:(n+1)]
    out[n+1    ,1:(n+1)] = out[1      ,1:(n+1)]
    out[1:(n+1),0      ] = out[1:(n+1),n      ]
    out[1:(n+1),n+1    ] = out[1:(n+1),1      ]
    out[0      ,0      ] = out[n      ,n      ]
    out[0      ,n+1    ] = out[n      ,1      ]
    out[n+1    ,n+1    ] = out[1      ,1      ]
    out[n+1    ,0      ] = out[1      ,n      ]
    return out

#----------------------------------------------------------------------------------------#
# Plot a hexagonal lattice from a matrix M
def plot_hex(M, clim=None, cmap=None, clabel=None, fig_name=None, title=None, tr=None, c=None, 
	     cbar=True, dpi=200):
    m = cp.copy(M)
    plt.rcParams.update({'font.size': 18}) 
    n = np.shape(m)[0]
    x = np.zeros((n,n))
    y = np.zeros((n,n))
    for i in range(n):
        y[i,:] = np.arange(1,n+1,1)
    for i in range(n):
        if i % 2 != 0:
            x[:,i] = np.arange(1,n+1,1)
        else:
            x[:,i] = np.arange(1,n+1,1) + 0.5

    fig, ax = plt.subplots(figsize=(11.6,10.0))
    if cmap==None:
        cmap = plt.cm.Spectral_r
        
    if tr==None:
        im = ax.scatter(x, y, c=m, s=700000/(n*n), cmap=cmap, linewidths=1, marker=(6, 0, 0))
    else:
        t = {}
        for i in np.arange(0,len(tr),1):
            t[i] = m<tr[i]
        m[m>np.max(tr)] = c[len(tr)]
        for i in np.arange(len(tr),0,-1):
            m[t[i-1]] = c[i-1]
        im = ax.scatter(x, y, c=m, s=700000/(n*n), cmap=cmap,linewidths=1, marker=(6, 0, 0))
        
    plt.xlim([1.0,n])
    plt.ylim([1.0,n])
    if title!=None:
        plt.title(title)  
    if clim!=None:
        im.set_clim(clim)
    if cbar:
        cax = fig.add_axes([0.91, 0.2, 0.04, 0.65])
	cbar = fig.colorbar(im, cax, orientation='vertical')
    if clabel!=None:
        cbar.set_label(clabel)
    if fig_name!=None:
        plt.savefig(fig_name, format='pdf', dpi=dpi)
    plt.show()


def plot_relativeLevel(eqs, p, v, r_v, key, pts_i=None, vlim=None, nsignal_dict=None, fig_name=None, clim=None,
                      show_snapshot=False, c=['b','r','g','m','k']):
    fs = {}
    for k in key:
        fs[k] = np.zeros(len(r_v))
    for i in range(len(r_v)):
        if v=='fng':
            p['ldf'] = 1.0 + 4.0*r_v[i]
            p['ljf'] = 1.0 - 0.9*r_v[i]
        elif v=='t':
            if i==0:
                p[v] = r_v[i]
            else:
                p[v] = r_v[i] - r_v[i-1] 
        else:
            p[v] = r_v[i]
        pts = euler_traj(eqs, p, pts=pts_i, vlim=vlim, nsignal_dict=nsignal_dict)
        if v=='t':
            pts_i = pts
        if show_snapshot:
            plot_hex(pts[key[0]], clim=clim,clabel=key)
        for k in key:
            fs[k][i] = np.mean(pts[k]) 
    
    plt.figure(figsize=(9,6))
    for k in range(len(key)):
        plt.plot(r_v, np.log2(fs[key[k]]/fs[key[k]][0]), 'go-', lw=2, ms=16, label=key[k], color=c[k])
    plt.xlabel(v)
    plt.ylabel('Relative level of proteins')    
    plt.legend()
    if fig_name!=None:
        plt.savefig(fig_name, format='pdf', dpi=200)
    plt.show()


def fractionStates(X, tr):
    N = np.float(np.size(X))
    if len(tr)==1:
        return np.sum(X<tr[0])/N, np.sum(X>tr[0])/N 
    elif len(tr)==2:
        return [np.sum(X<tr[0])/N, np.sum((X>tr[0]) & (X<tr[1]))/N, np.sum(X>tr[1])/N]
    
def plot_fractionStates(eqs, p, v, r_v, key, tr, pts_i=None, vlim=None, l=['M','E/M','E'], c=['#e8656c','#e4fc36','#00ff9c'], 
			show_snapshot=False, nsignal_dict={'N': ['D', 'J'], 'I': ['D', 'J'], 'D': ['N'], 'J': ['N']}, fig_name=None):
    fs = []
    for i in range(len(r_v)):
        if v=='fng':
            p['ldf'] = 1.0 + 4.0*r_v[i]
            p['ljf'] = 1.0 - 0.9*r_v[i]
        elif v=='t':
            if i==0:
                p[v] = r_v[i]
            else:
                p[v] = r_v[i] - r_v[i-1] 
        else:
            p[v] = r_v[i]
        pts = euler_traj(eqs, p, pts=pts_i, vlim=vlim, nsignal_dict=nsignal_dict)
        if v=='t':
            pts_i = pts
        if show_snapshot:
            plot_hex(pts[key], clim=clim, clabel=key)
        fs += [fractionStates(pts[key],tr[key])]
    fs = np.array(fs)
    
    fig = plt.figure(figsize=(9,6))
    colors=['#e8656c','#e4fc36','#00ff9c']
    for i in range(len(tr[key])+1):
        plt.plot(r_v, fs[:,i], ['o-','>-','s-'][i], c=c[i], lw=2, ms=16, label=l[i])
    plt.xlabel(v)
    plt.ylabel('Fraction states')    
    plt.legend()
    if fig_name!=None:
        plt.savefig(fig_name, format='pdf', dpi=200)
    plt.show()
#----------------------------------------------------------------------------------------#    



#-------------------------------------------------------------------------------------------------------------------#
def eliminate_redundants(fp, eps=10):
    for i in range(len(fp)):
        for k, v in fp[i].items():
            v = round(v,eps) 
            fp[i][k] = v
    seen = set()
    new_l = []
    for d in fp:
        t = tuple(d.items())
        if t not in seen:
            seen.add(t)
            new_l.append(d)
    return new_l
  
#-------------------------------------------------------------------------------------------------------------------#
def stability(FPs, ODE, eps=0.1):
    out = []
    for i in range(len(FPs)):
        X = {}
        stable = True
        for k in FPs[0].keys():
            X[k] = FPs[i][k]*(1 + eps*rand.sample(list([-1,1]),1)[0])
        ODE.set(ics  = X)  
        traj = ODE.compute('traj')
        X = traj.sample()[-1]
        for k in FPs[0].keys():
            if np.abs(X[k]-FPs[i][k]) > eps*FPs[i][k]:
                stable = False
        out += ['S'] if stable else ['I']
    return out

#-------------------------------------------------------------------------------------------------------------------#
def PyCont_args(nmodel, freepar, maxnumpoints, maxstep=1e+1, minstep=1e-1, stopAt=['B'],
                step=1e-0, LocBifPoints=['BP','LP','B'], saveeigen=False, Type='EP-C'):
    PCargs = PyDSTool.args(name=nmodel, type=Type)    # 'EP-C' stands for Equilibrium Point Curve.
    PCargs.freepars     = [freepar]                   # control parameter 
    PCargs.MaxNumPoints = maxnumpoints                # The following 3 parameters are set after trial-and-error
    PCargs.MaxStepSize  = maxstep
    PCargs.MinStepSize  = minstep
    PCargs.StepSize     = step
    PCargs.StopAtPoints = stopAt    
    PCargs.LocBifPoints = LocBifPoints                # detect limit points / saddle-node bifurcations
    PCargs.SaveEigen    = saveeigen                   # to tell unstable from stable branches
    return PCargs
 
#-------------------------------------------------------------------------------------------------------------------#
#   Plot functions
#-------------------------------------------------------------------------------------------------------------------#
def hist_clustersize(dic, keys, tr, clim, norm=True, bars=False, fig_name=None, higher=True):
    plt.figure(figsize=(7,5), dpi=200)
    for k in keys:
        h = np.zeros((len(dic),clim[1]-clim[0]+1))
        for j in range(len(dic)):
            x = np.array(dic[j][k])
            if higher:
	        x[x<tr[k][0]] = 0.0
                x[x>tr[k][0]] = 1.0
            #else:
	        x[x<tr[k][0]] = 1.0
                x[x>tr[k][0]] = 0.0
            xs, n_clusters = measurements.label(x)
            print n_clusters, 'clusters found'
            a  = measurements.sum(x, xs, index=arange(xs.max() + 1))
            ma = np.max(a) 
            h[j,:] = np.asarray([np.sum(a==i) for i in np.arange(clim[0],clim[1]+1,1)])
        hm = np.mean(h,axis=0)
        if norm:
            hm = hm/np.sum(hm)
            plt.ylim([0.0,1.0])
        if bars:
            plt.bar(np.arange(clim[0]-0.4, clim[1], 1.0 ),hm)
        else:
            plt.plot(np.arange(clim[0], clim[1]+1, 1.0 ),hm, 'ob-', ms=10)
        plt.xlim([clim[0]-0.5,clim[1]+0.5])
        plt.xticks(range(clim[0], clim[1]+1,1))
        plt.xlabel('Cluster size')
        plt.ylabel('Number of clusters')
        #plt.title(k)
        if fig_name!=None:
            plt.savefig(fig_name, format='pdf', dpi=200)
            
 

#-------------------------------------------------------------------------------------------------------------------#
def plot_fates(dic, keys, tr, colors=['#e8656c','#e4fc36','#00ff9c'], vlim=[0.0, 1.0], ncomb=1, fontsize=16, fig_name=None):
    plt.figure(figsize=(8,7), dpi=200)
    cmap = cl.ListedColormap(colors)
    for k in keys:
        for j in range(len(dic)):
            x = np.array(dic[j][k])
            if len(tr[k])==1:
                x[x < tr[k][0]] = 0.0
                x[x > tr[k][0]] = 1.0
            elif len(tr[k])==2:
                x[x < tr[k][0]] = 0.0
                x[x > tr[k][1]] = 1.0
                x[(x<tr[k][1]) & (x>tr[k][0])] = 0.5
    plt.pcolor(x, edgecolors='k', linewidths=1, cmap=cmap, vmin=vlim[0], vmax=vlim[1])
    if fig_name!=None:
        plt.savefig(fig_name, format='pdf', dpi=200)

#-------------------------------------------------------------------------------------------------------------------#
def plot_pcolors(dic, keys, fs=[10,7], ncol=None, nrow=None, fontsize=12, fig_name=None):
    if ncol == None:
        ncol = len(keys)
    if nrow == None:
        nrow = 1
    plt.figure(figsize=(fs[0]*ncol,fs[1]*nrow), dpi=200)  
    for k in range(len(keys)):
        plt.subplot(nrow,ncol,k+1)
        plt.title(keys[k])
        plt.pcolor(np.asarray(dic[0][keys[k]]))
        plt.colorbar()
    if fig_name!=None:
        plt.savefig(fig_name, format='pdf', dpi=200)
#-------------------------------------------------------------------------------------------------------------------#
def hist_dist(dic, key, hr, tr=None, a=None, fig_name=None, nbins=20, bar_width=1, bar=False, c='b', m='-o', leg=False):
    tr_key = tr.keys()[0]
    h = np.zeros((len(dic),nbins))
    bar_width = (hr[key][1] - hr[key][0])/float(nbins)
    for j in range(len(dic)):
        x = np.array(dic[j][key])
        if a==None or tr==None:
	    h[j,:] = np.histogram(x, range=hr[key], bins=nbins)[0]
        elif a==-1:
            h[j,:] = np.histogram(x[x<tr[tr_key][0]], range=hr[key], bins=nbins)[0]
        elif a==0:
            h[j,:] = np.histogram(x[(x>tr[tr_key][0]) & (x<tr[tr_key][1])], range=hr[key], bins=nbins)[0]
        elif a==+1:
            h[j,:] = np.histogram(x[x>tr[tr_key][1]], range=hr[key], bins=nbins)[0]
	    
    hm = np.mean(h,axis=0)
    if bar:
        plt.bar(np.arange(hr[key][0],hr[key][1], bar_width), hm/np.sum(hm), bar_width)
    else:
        plt.plot(np.arange(hr[key][0],hr[key][1], bar_width), hm/np.sum(hm), m, color=c, ms=8, mew=2)
    plt.xlim(hr[key])
    plt.xlabel('amount of protein')
    plt.ylabel('fraction of cells')
    plt.legend(key)
    if fig_name!=None:
        plt.savefig(fig_name, format='pdf', dpi=200)
        
#-------------------------------------------------------------------------------------------------------------------#
def plot_continuation(ODE, freepar, keys, bif_startpoint, LocBifPoints=['LP','B'], PCargs=None, returnLP=None, ics=None, 
		      xlim=None, ylim=None, xticks=False, yticks=False, maxstep=1e+2, minstep=1e-2, step=5e+1, maxpoints=500, 
		      off_points=True, nrow=None, ncol=None, showcurve=True, n_form_coef=False, silence=False, 
		      fs=[6,5], fontsize=18, fig_name=False):
    plt.rcParams.update({'font.size': fontsize}) 
    ODE.set(pars = {freepar: bif_startpoint})

    if silence:
        class NullDevice():
            def write(self, s):
                pass
        original_stdout = sys.stdout
        sys.stdout = NullDevice()
    if showcurve:
        if ncol == None:
            ncol = len(keys)
        if nrow == None:
            nrow = 1
        plt.figure(figsize=(fs[0]*ncol,fs[1]*nrow), dpi=200)
    if ics==None:
        ics = [eliminate_redundants(pp.find_fixedpoints(ODE, n=2, maxsearch=1e+4, eps=1e-12), 4)[0]]
    if PCargs==None:
        PCargs = PyCont_args(ODE.name, freepar, maxpoints, saveeigen=True, LocBifPoints=LocBifPoints, 
			     maxstep=maxstep, minstep=minstep, step=step) 
 
    for j in range(len(ics)):
        ODE.set(ics  = ics[j])
        PyCont = PyDSTool.ContClass(ODE)     
        PyCont.newCurve(PCargs)
        PyCont[ODE.name].forward()
        PyCont[ODE.name].backward()
        if showcurve:        
            for i in range(len(keys)):
                PyCont.display((freepar,keys[i]), stability=True, axes=(nrow,ncol,i+1), color='k', linewidth=3)
                if off_points:
                    PyCont.plot.toggleLabels('off')
                plt.xlabel(freepar, fontsize=18)
                plt.ylabel(keys[i], fontsize=18)
                plt.title('')
                if xlim != None:
                    plt.xlim([xlim[0],xlim[1]])
                if ylim != None:
                    plt.ylim([ylim[0],ylim[1]])
                if xticks:
                    plt.xticks(xticks)
                if yticks:
                    plt.yticks(yticks)
    if n_form_coef:
        i = 1
        while PyCont[ODE.name].getSpecialPoint('LP'+str(i)):
            print("LP"+str(i), PyCont[ODE.name].getSpecialPoint('LP'+str(i)).labels['LP']['data'])
            i += 1
    if fig_name:
        plt.savefig(fig_name, format='pdf', dpi=200)
    if silence:
        sys.stdout = original_stdout
    if returnLP!=None:
        P = []
        for k in returnLP:
            i=1
            while PyCont[ODE.name].getSpecialPoint(k+str(i)):
                P += [PyCont[ODE.name].getSpecialPoint(k+str(i))[freepar]]
                i +=1
        return P
    plt.show()

#-------------------------------------------------------------------------------------------------------------------#
def plot_continuation_twocells(ODE, freepar, keys, bif_startpoint, LocBifPoints=['LP','BP','B'], xlim=False, ylim=False, 
			       maxpoints=500, fig_name=False, saveeigen=True, maxstep=1e-0, minstep=1e-2, step=5e-2, silence=False):
    if silence:
        class NullDevice():
            def write(self, s):
                pass
        original_stdout = sys.stdout
        sys.stdout = NullDevice()
    fp = eliminate_redundants(pp.find_fixedpoints(ODE, n=2, maxsearch=1e+4, eps=1e-10),6)
    p = plot_continuation(ODE, freepar, keys, bif_startpoint, ics=[fp[0]], maxstep=maxstep, minstep=minstep, 
                          step=step, off_points=True,  LocBifPoints=LocBifPoints, returnLP='BP')
    figure(figsize=(6*len(keys),5))
    l = sorted(p + ODE.pdomain[freepar])
    for i in range(len(p)):
        plt.axvline(x=p[i],color='k')
    
    for i in range(len(l)-1):
        ODE.set(pars = {freepar: (l[i] + l[i+1])/2} )
        fp = eliminate_redundants(pp.find_fixedpoints(ODE, n=2, maxsearch=1e+4, eps=1e-10),6)
        print(len(fp), fp)
        PCargs = PyCont_args(ODE.name, freepar, maxpoints, stopAt=['B','BP'], maxstep=maxstep, minstep=minstep, step=minstep)
        for j in range(len(fp)):
            ODE.set(ics  = fp[j])
            PyCont = PyDSTool.ContClass(ODE)     
            PyCont.newCurve(PCargs)
            PyCont[ODE.name].forward()
            PyCont[ODE.name].backward()
            for k in range(len(keys)):
                PyCont.display((freepar,keys[k]), axes=(1,len(keys),k+1), stability=True, linewidth=2)
                PyCont.plot.toggleLabels('off')
                plt.title('')
                plt.xlim(xlim)
                if ylim:
                    plt.ylim(ylim)
    if fig_name:
        plt.savefig(fig_name, format='pdf', dpi=200)
    if silence:
        sys.stdout = original_stdout
  
#-------------------------------------------------------------------------------------------------------------------#
def plot_phasediagram(ODE, freepar, v, r_v, bif_startpoint, keys=False, xlim=False, ylim=False, xticks=False, yticks=False, 
		      show_continuation=False, maxstep=1e+2, minstep=1e-2, step=5e+1, maxpoints=500, nrow=None, ncol=None,
		      LocBifPoints=['LP','B'], BifPoints=['LP'], silence=False, fig_name=False, fast_fp=False, returnLPs=False):
    if silence:
        class NullDevice():
            def write(self, s):
                pass
        original_stdout = sys.stdout
        sys.stdout = NullDevice()
    if keys==False:
        keys = ODE.variables.keys()
        
    x = []
    for i in r_v:
        ODE.set(pars = {v: i, freepar: bif_startpoint})
        if fast_fp:
	    fp = fast_fixedpoint(ODE)
	else:
	    fp = eliminate_redundants(pp.find_fixedpoints(ODE, n=2, maxsearch=1e+4, eps=1e-12),4)[0]
        ODE.set(ics  = fp)        
        PCargs = PyCont_args(ODE.name, freepar, maxpoints, saveeigen=True, maxstep=maxstep, minstep=minstep, step=step, LocBifPoints=LocBifPoints) 
        lp = plot_continuation(ODE, freepar, keys, bif_startpoint, PCargs=PCargs, returnLP=BifPoints, showcurve=show_continuation, ics=[fp],
			       fs=[4,3], fontsize=12,nrow=nrow, ncol=ncol)
	x += [lp]
        plt.show()
    if silence:
        sys.stdout = original_stdout

    figure(figsize=(6,5), dpi=200)
    x = np.asarray(x)
    for i in range(shape(x)[1]):
        plot(x[:,i], r_v, color='k')
    
    plt.xlabel(freepar, fontsize= 18)
    plt.ylabel(v, fontsize= 18)
    if xlim:
        plt.xlim(xlim)
    if xticks:
        plt.xticks(xticks)
    if ylim:
        plt.ylim(ylim)
    if yticks:
        plt.yticks(yticks)
    if fig_name:
        plt.savefig(fig_name, format='pdf', dpi=200)
    plt.show()
    if returnLPs:
	return x
    
#-------------------------------------------------------------------------------------------------------------------#
def nullclines(axis, DSargs, stab, fp, nfp=0, vlim=None, c = ['b','g'], maxpoints=[1000,1000], step=5e+1, minstep=1e-1, 
	       maxstep=1e+3, fs=[6,5], fig_name=False, plotaxis=[0,1], loc=0, fontsize=18, pcontour=None, silence=False,
	       xticks=False, yticks=False):
    plt.rcParams.update({'font.size': fontsize}) 
    if silence:
        class NullDevice():
            def write(self, s):
                pass
        original_stdout = sys.stdout
        sys.stdout = NullDevice()

    figure(figsize=(fs[0],fs[1]), dpi=200)
    DSnc = cp.deepcopy(DSargs) 
    for i in plotaxis:
        keys = list(DSargs.varspecs.keys())
        keys.remove(axis[i])
        DSnc.pars[axis[i]] = fp[nfp][axis[i]]
        DSnc.varspecs = {}
        DSnc.ics = {}
        DSnc.xdomain = {}
        DSnc.pdomain = {}
        if vlim != None:
            DSnc.pdomain[axis[i]] = vlim[axis[i]]
        for k in keys:
            DSnc.varspecs[k] = DSargs.varspecs[k]
            DSnc.ics[k]      = fp[nfp][k]
            DSnc.xdomain[k]  = DSargs.xdomain[k]
        ODEnc = Vode_ODEsystem(DSnc) 
        PCargs = PyCont_args('nullclines', axis[i], maxpoints[i], maxstep=maxstep, minstep=minstep, step=step, LocBifPoints=['B'])
        PCargs.StopAtPoints = ['B']
        PyCont = PyDSTool.ContClass(ODEnc)
        PyCont.newCurve(PCargs)
        PyCont['nullclines'].forward()
        PyCont['nullclines'].backward()
        PyCont.display((axis[0],axis[1]), stability=True, linewidth=3, color=c[i], label='d'+axis[i]+'/dt'+' != 0' )
        PyCont.plot.toggleLabels('off')
        PyCont.plot.togglePoints('off')
        del DSnc.pars[axis[i]]

    for i in range(len(fp)):
        plt.plot(fp[i][axis[0]],fp[i][axis[1]], 'ok', markersize=12, markerfacecolor='r' if stab[i]=='S' else 'w')
        
    if pcontour!=None:  
        H, xedges, yedges = np.histogram2d(asarray(pcontour[axis[0]])[:,0], asarray(pcontour[axis[1]])[:,0], bins=100)
        H = np.rot90(H)
        H = np.flipud(H)
        xbin = 0.5*(xedges[1] - xedges[0])
        ybin = 0.5*(yedges[1] - yedges[0])
        plt.contour(xedges[1:]-xbin, yedges[1:]-ybin, H)
    plt.xlabel(axis[0])
    plt.ylabel(axis[1])
    plt.title('')
    plt.legend(loc=loc)
    if vlim != None:
        plt.xlim((vlim[axis[0]][0],vlim[axis[0]][1]))
        plt.ylim((vlim[axis[1]][0],vlim[axis[1]][1]))
    if xticks:
        plt.xticks(xticks)
    if yticks:
        plt.yticks(yticks)
    if fig_name:
        plt.savefig(fig_name, format='pdf', dpi=200)
    if silence:
        sys.stdout = original_stdout
        
#-------------------------------------------------------------------------------------------------------------------#
def param_sensitivity_bars(list_pars, ODE, DSargs, var, fig_name=False, fs=[10,5], delta=[0.0, 0.1, -0.1]):
    change = {}
    for pars in list_pars:
        if DSargs.pars[pars] != 0:
            a = []
            for d in delta:
                ODE.set(pars = {pars: (1.0 + d)*DSargs.pars[pars]} ) 
                a += [eliminate_redundants(pp.find_fixedpoints(ODE, n=2, maxsearch=1e+4, eps=1e-12),6)[0][var]]
            change[pars] = [100*(a[2] - a[0])/a[0],100*(a[1] - a[0])/a[0]] 
        else:
            change[pars] = [0,0] 
    l = change.keys()
    isort = np.argsort([np.abs(change[i][0])+np.abs(change[i][1]) for i in l])[::-1]

    figure(figsize=(fs[0],fs[1]), dpi=200)
    plt.bar(range(len(change.keys())), [change[l[i]][0] for i in isort], color='r', align='center', alpha=0.8)
    plt.bar(range(len(change.keys())), [change[l[i]][1] for i in isort], color='b', align='center', alpha=0.8)
    plt.xticks(np.arange(len(list_pars)+1), [l[i] for i in isort])
    plt.xlim([-1,len(list_pars)])
    plt.ylabel('Change in the signal (%)', fontsize= 18)
    plt.legend( ('- 10%', '+10%'), loc='upper right')
    if fig_name:
        plt.savefig(fig_name, format='pdf', dpi=200) 
    plt.show()

#-------------------------------------------------------------------------------------------------------------------#
def param_sensitivity_bifurcations(DSargs, freepar, key, list_pars, bif_startpoint, d=[0.0, 0.1, -0.1], c=['k', 'b', 'r'], 
                                   ylim=False, xlim=False, xticks=False, yticks=False, fig_name=False, ncol=False, 
                                   nrow=False, maxstep=1e+2, minstep=1e-2, step=5e+1, silence=False):
    if silence:
        class NullDevice():
            def write(self, s):
                pass
        original_stdout = sys.stdout
        sys.stdout = NullDevice()
    if ncol==False:
        nrow=1
        ncol=len(list_pars)
    else:
        plt.figure(figsize=(6*ncol,5*nrow), dpi=200)
    for i in range(len(list_pars)):
        ODE = Vode_ODEsystem(DSargs)
        for j in range(len(d)):
            ODE.set(pars = {list_pars[i]: (1.0 + d[j])*DSargs.pars[list_pars[i]]} ) 
            fp_coord = eliminate_redundants(pp.find_fixedpoints(ODE, n=2, maxsearch=1e+4, eps=1e-10),6)
            ODE.set(ics  = fp_coord[0])  
            PCargs = PyCont_args('psensit', freepar, 200, saveeigen=True, maxstep=maxstep, minstep=minstep, step=step,
				 LocBifPoints=['B','LP'])
            PyCont = PyDSTool.ContClass(ODE)     
            PyCont.newCurve(PCargs)
            PyCont['psensit'].forward()
            PyCont['psensit'].backward()
            
            PyCont.display((freepar,key), stability=True, axes=(nrow,ncol,i+1), color=c[j], linewidth=3)
            plot(0, 0, linewidth=3, color=c[j])
            PyCont.plot.toggleLabels('off')
        plt.title(list_pars[i])
        if i == 0:
            plt.legend(('0%','+10%','- 10%'))
        if ylim:
            plt.ylim(ylim)
        if ylim:
            plt.ylim(ylim)
        if xticks:
            plt.xticks(xticks)
        if yticks:
            plt.yticks(yticks)
    if fig_name:
        plt.savefig(fig_name, format='pdf', dpi=200) 
    if silence:
        sys.stdout = original_stdout

#-------------------------------------------------------------------------------------------------------------------#
def plot_3Dpotential(x1, x2, npoints, xlim=False, ylim=False, zlim=False, offset=5, cut=9.5, fig_name=False, 
		     nbins=100, scale=1000.0):
    plt.rcParams.update({'font.size': 22}) 
    j = np.isfinite(x1) & np.isfinite(x2)

    x = x1[j]
    y = x2[j]
    H, xedges, yedges = np.histogram2d(x, y, bins=nbins)

    H = np.rot90(H)
    H = np.flipud(H)
    V = -np.log(H/npoints)
    V[V==np.inf]=float('NaN')
    V[V>cut]=float('NaN')
    Z = V
    
    X,Y = np.meshgrid(xedges[1:]/scale, yedges[1:]/scale)

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    cset = ax.contour(X, Y, Z, zdir='z', offset=offset, cmap=cm.coolwarm)
    pc = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.4, cmap=cm.coolwarm, linewidth=0.1)
    pc.set_clim(np.min(V[V>0]),np.max(V[V>0]))

    ax.view_init(45,-135)
    if xlim:
        ax.set_xlim3d(xlim)
    if ylim:
        ax.set_ylim3d(ylim)
    if zlim:
        ax.set_zlim3d(zlim)
    ax.zaxis.set_rotate_label(False)  # disable automatic rotation
    ax.set_zlabel('Effective Potential', rotation=95)
    if fig_name:
        plt.savefig(fig_name+'.png', format='png', dpi=200)
        plt.savefig(fig_name+'.pdf', format='pdf', dpi=200)
    plt.show()

#----------------------------------------------------------------------------------------#
def plot_trajectory(ODE, keys, dt=None, t=None, linewidth=2, fontsize=18):
    traj = ODE.compute('test_traj')
    pts = traj.sample()
    plt.rcParams.update({'font.size': fontsize}) 
    plt.plot(pts[keys[0]], pts[keys[1]], color='k', linewidth=linewidth)
    plt.xlabel(keys[0])
    plt.ylabel(keys[1])
    plt.show()
#----------------------------------------------------------------------------------------#   
def dist2D(dic, keys, tr={}, fig_name=None, leg=False):
    plt.figure(figsize=(6,5), dpi=200)
    for j in range(len(dic)):
        w = np.array(dic[j]['W'])
        if len(keys[0]) > 1:
	    x = np.array(dic[j][keys[0][0]]) + np.array(dic[j][keys[0][1]])
	else:
	    x = np.array(dic[j][keys[0]])
        y = np.array(dic[j][keys[1]])
        plt.plot(x[w<tr['W'][0]]                   , y[w<tr['W'][0]]                   , 'o', c='#e8656c')
        plt.plot(x[(w>tr['W'][0]) & (w<tr['W'][1])], y[(w>tr['W'][0]) & (w<tr['W'][1])], 'o', c='#e4fc36')
        plt.plot(x[w>tr['W'][1]]                   , y[w>tr['W'][1]]                   , 'o', c='#00ff9c')
        plt.xlabel(keys[0])
        plt.ylabel(keys[1])
        #if lim != None:
	    #if lim[]
        if leg:
            plt.legend(leg)
        if fig_name!=None:
            plt.savefig(fig_name, format='pdf', dpi=200)
            
#----------------------------------------------------------------------------------------#   
def hist_dist(dic, key, hr, tr={}, a=None, fig_name=None, nbins=10, bar_width=1, bar=False, 
              c='b', m='-o', leg=False):
    
    tr_key = tr.keys()[0]
    h = np.zeros((len(dic),nbins))
    bar_width = (hr[key][1] - hr[key][0])/float(nbins)
    for j in range(len(dic)):
        x = np.array(dic[j][key])
        y = np.array(dic[j][tr_key])
        if a==None:
            h[j,:] = np.histogram(x, range=hr[key], bins=nbins)[0]
        elif a==-1:
            h[j,:] = np.histogram(x[y<tr[tr_key][0]], range=hr[key], bins=nbins)[0]
        elif a==0:
            h[j,:] = np.histogram(x[(y>tr[tr_key][0]) & (y<tr[tr_key][1])], range=hr[key], bins=nbins)[0]
        elif a==+1:
            h[j,:] = np.histogram(x[y>tr[tr_key][1]], range=hr[key], bins=nbins)[0]
  
    hm = np.mean(h,axis=0)
    print hm, a
    if np.sum(hm)>0.0:
        if bar:
            plt.bar(np.arange(hr[key][0],hr[key][1], bar_width), hm/np.sum(hm), bar_width, color=c)
        else:
            plt.plot(np.arange(hr[key][0],hr[key][1], bar_width), hm/np.sum(hm), m, color=c, ms=12, mew=2)
        plt.xlim(hr[key])
        plt.xlabel('amount of protein')
        plt.ylabel('fraction of cells')
        plt.title(key)
        if fig_name!=None:
            plt.savefig(fig_name, format='pdf', dpi=200)
            

#-------------------------------------------------------------------------------------------------------------------#
def fast_fixedpoint(ODE, tdomain=[0, 100000]):
    ODE.set(tdomain=tdomain)
    traj = ODE.compute('traj')
    pts = traj.sample()
    return dict(pts[-1])