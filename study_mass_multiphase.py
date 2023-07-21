""" Finite difference method for P phases """

# Modules

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import animation , rc
import random as rd
import seaborn as sns
import pandas as pd


if __name__== "__main__":
    
    # Parameters of our scheme
    P = 4 # number of phases
    T = 300 # final time
    X = 25 # maximum age such that we don't miss information in true differential system
    List_Kmax = [2 for p in range(P)] # List of maximum values of the kernel
    Kmax = max(List_Kmax)
    delta_x_wanted = 1/10 # Age step wanted
    Parameters_initial = [[3+p,1] for p in range(P)]
    # If each initial conditions are Gaussian, we need to specify the mean and the standard deviation
    Parameters_kernel = [[8,8,10+p,24] for p in range(P)]
    # If each kernel is an approximation a a simple function independant of time, 
    # we need a slope to the left, a slope to the right, a minimum age where transition starts
    # and a maximum age where the transition ends
    List_Is_zero = [False for p in range(P)]
    fps1 = 8 # Number of frame per second for the animations
    name = "Test" # Name for the animation
    memo = True # If we save the animation
    

def delta(T,X,delta_x_wanted,Kmax):
    """
    CFL conditions :
    We want Delta_t < Delta_x /(1 + Kmax Delta_x)

    Parameters
    ----------
    T : float
        Final time of the scheme.
    X : float
        Maximum age of the scheme.
    delta_x_wanted : float
        Age step wanted.
    Kmax : float
        Upper bound of the kernel.
        
        
    Returns
    -------
    delta_t : float
        Time step.
    delta_x : float
        Age step.
    I : int
        I+1 number of element in Time.
    J : int
        J+1 number of element in Age.
    Time : list of float
            Discretisation of time.
    Age : list of float
            Discretisation of age.
    """
    
    delta_t =(delta_x_wanted/(1+Kmax*delta_x_wanted))/1.01
    I = int(T/delta_t)
    J = int(X/delta_x_wanted)
    # the choice of I and J impose that the CFL conditions are still true
    Time = np.array([i*delta_t for i in range(I+1)]) # discretisation of our time
    Age = np.array([j*delta_x_wanted for j in range(J+1)]) # discretisation of our age
    
    return(delta_t,delta_x_wanted,I,J,Time,Age)


if __name__== "__main__":
    delta_t,delta_x,I,J,Time,Age = delta(T,X,delta_x_wanted,Kmax)
    
    
    
    
    
def initialn_p(x,p,Parameters_initial,Is_zero):
    if Is_zero :
        return(0)
    else :
        mu,sigma = Parameters_initial[p]
        result = np.exp(-(x-mu)**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)
        return(result)

def graph_initial(Age,Parameters_initial,List_Is_zero):
    P = len(Parameters_initial)
    Y = [[] for p in range(P)]
    for p in range(P):
        Is_zero_p = List_Is_zero[p]
        Y[p] += [initialn_p(x,p,Parameters_initial,Is_zero_p) for x in Age]
        plt.plot(Age,Y[p],label = "Phase %s"%(p+1))
    plt.legend()
    plt.show()
    
    
    
    
def kernel_p(t,x,p,List_Kmax,Parameters_kernel): 
    Kmaxp = List_Kmax[p]
    sl,sr,xmin,xmax = Parameters_kernel[p]
    coeff = 1+p/10
    result = (np.tanh(sl*(x-(xmin+coeff*np.cos(p*t)))) + np.tanh(sr*(xmax-x)))*Kmaxp/2
    return(result)


def graph_kernel(Age,List_Kmax,Parameters_kernel): # kernel supposed independant of time
    P = len(Parameters_kernel)
    Y = [[] for p in range(P)]
    for p in range(P):
        Y[p] += [kernel_p(0,x,p,List_Kmax,Parameters_kernel) for x in Age]
        plt.plot(Age,Y[p],label="Kernel for p=%s"%(p+1))
    plt.legend()
    plt.show()
    
    
def separation(Vector,P):
    """
    Vector is a concatenation of P subvectors, the function separates them

    Parameters
    ----------
    Vector : array
        Concatenation of P subvectors.
    P : int
        Number of subvectors.

    Returns
    -------
    Subvectors : array of list of float
        List of subvectors.
    """
    lenght = int(np.size(Vector)/P)
    Subvectors = []
    for p in range(P):
        sub = Vector[p*lenght:(p+1)*lenght]
        Subvectors += [sub]
    Subvectors = np.array(Subvectors)
    return(Subvectors)
        
    
def matrix_transition(t,Age,delta_t,delta_x,kernel_p,List_Kmax,Parameters_kernel):
    """
    Creation of the matrix of transition between t_(i) and t_(i+1)

    Parameters
    ----------
    t : float
        Time of the transition.
    Age : list of float
        List of the discretise age space.
    delta_t : float
        Time step.
    delta_x : float
        Age step.
    kernel_p : function
        Definition of the kernel (with P phases).
    List_Kmax : list of float
        List of maximum value of each kernel.
    Parameters_kernel : list of list of float
        List of parameters of each kernel.

    Returns
    -------
    mtx : array
        Matrix of transition at time t.

    """
    
    
    L = np.size(Age)
    P = len(List_Kmax)
    ratio = delta_t/delta_x
    Row = []
    Col = []
    Data = []
    # if p = 0 (the first phase)
    Row += [0 for l in range(L)]
    Col += [(P-1)*L+l for l in range(L)]
    Data += [(delta_x/2)*kernel_p(t,Age[0],P-1,List_Kmax,Parameters_kernel)]
    Data += [delta_x*kernel_p(t,Age[j],P-1,List_Kmax,Parameters_kernel) for j in range(1,L-1)]
    Data += [(delta_x/2)*kernel_p(t,Age[-1],P-1,List_Kmax,Parameters_kernel)]
    
    Row += [l for l in range(1,L)]
    Col += [l for l in range(L-1)]
    Data += [ratio for l in range(L-1)]
    
    Row += [l for l in range(1,L)]
    Col += [l for l in range(1,L)]
    Data += [1-ratio-delta_t*kernel_p(t,Age[l],0,List_Kmax,Parameters_kernel) for l in range(1,L)]
    
    
    for p in range(1,P):
        Row += [p*L for l in range(L)]
        Col += [(p-1)*L+l for l in range(L)]
        Data += [(delta_x/2)*kernel_p(t,Age[0],p-1,List_Kmax,Parameters_kernel)]
        Data += [delta_x*kernel_p(t,Age[j],p-1,List_Kmax,Parameters_kernel) for j in range(1,L-1)]
        Data += [(delta_x/2)*kernel_p(t,Age[-1],p-1,List_Kmax,Parameters_kernel)]
        
        Row += [l+p*L for l in range(1,L)]
        Col += [l+p*L for l in range(L-1)]
        Data += [ratio for l in range(L-1)]
        
        Row += [l+p*L for l in range(1,L)]
        Col += [l+p*L for l in range(1,L)]
        Data += [1-ratio-delta_t*kernel_p(t,Age[l],p,List_Kmax,Parameters_kernel) for l in range(1,L)]
        
    row = np.array(Row)
    col = np.array(Col)
    data = np.array(Data)
    mtx = sp.sparse.coo_matrix((data, (row, col)), shape=(L*P, L*P))
    
    return(mtx)

    
def heatmap(t,Age,delta_t,delta_x,kernel_p,List_Kmax,Parameters_kernel):
    """
    Heatmap representation of the matrix of transition at time t

    Parameters
    ----------
    t : float
        Time of the transition.
    Age : list of float
        List of the discretise age space.
    delta_t : float
        Time step.
    delta_x : float
        Age step.
    kernel_p : function
        Definition of the kernel (with P phases).
    List_Kmax : list of float
        List of maximum value of each kernel.
    Parameters_kernel : list of list of float
        List of parameters of each kernel.

    Returns
    -------
    heatmap : graph
        Heatmap of the matrix of transition at time t.

    """
    plt.close()
    mtx = matrix_transition(t,Age,delta_t,delta_x,kernel_p,List_Kmax,Parameters_kernel)
    Mtx = mtx.todense()
    plt.gcf().suptitle("Matrix of transition at time t=%s with %s phases"%(t,P))
    heatmap = sns.heatmap(Mtx)
    return(heatmap)




def step(t,Density_old,Age,delta_t,delta_x,kernel_p,List_Kmax,Parameters_kernel):
    """
    Calculation of the approximated density of each phase at time t_(i+1)

    Parameters
    ----------
    t : float
        Time of the transition.
    Density_old : array of float
        Previous densities (concatenation of density approximation in each phase)
    Age : list of float
        List of the discretise age space.
    delta_t : float
        Time step.
    delta_x : float
        Age step.
    kernel_p : function
        Definition of the kernel (with P phases).
    List_Kmax : list of float
        List of maximum value of each kernel.
    Parameters_kernel : list of list of float
        List of parameters of each kernel.

    Returns
    -------
    Density : array of float
        New densities (concatenation of density approximation in each phase)

    """
    M = matrix_transition(t,Age,delta_t,delta_x,kernel_p,List_Kmax,Parameters_kernel)
    Density = M@Density_old
    return(Density)


def animation_density(T,X,P,delta_x_wanted,initialn_p,kernel_p,List_Is_zero,List_Kmax,Parameters_kernel,fps1,name,memo):
    """
    Animation of the first 4 densities across time (thus imposing P>=4)

    Parameters
    ----------
    T : float
        Maxmimum time.
    X : float
        Maximum age.
    P : int
        Number of phases.
    Delta_x_wanted : float
        Age step wanted.
    initialn_p : function
        Definition of the initial conditions (with P phases).
    kernel_p : function
        Definition of the kernel (with P phases).
    List_Is_zero : list of boolean
        If the initial density of phase p is null, then List_Is_zero[p] = True , else it is False
    List_Kmax : list of float
        List of maximum value of each kernel.
    Parameters_kernel : list of list of float
        List of parameters of each kernel.
    fps1 : int
        Number of FPS for the animation.
    name : char
        Name of the GIF.
    memo : bool
        True if we want to save the animation, False otherwise.

    Returns
    -------
    Animation of the first four phases

    """
    Kmax = max(List_Kmax)
    Kmax1,Kmax2,Kmax3,Kmax4 = List_Kmax[0:4]
    mu1,sigma1 = Parameters_initial[0]
    mu2,sigma2 = Parameters_initial[1]
    mu3,sigma3 = Parameters_initial[2]
    mu4,sigma4 = Parameters_initial[3]
    sl1,sr1,xmin1,xmax1 = Parameters_kernel[0]
    sl2,sr2,xmin2,xmax2 = Parameters_kernel[1]
    sl3,sr3,xmin3,xmax3 = Parameters_kernel[2]
    sl4,sr4,xmin4,xmax4 = Parameters_kernel[3]
    
    delta_t,delta_x,I,J,Time,Age = delta(T,X,delta_x_wanted,Kmax)
    L = J+1
    
    Density = []
    P = len(Parameters_initial)
    for p in range(P):
        Is_zero_p = List_Is_zero[p]
        Density += [initialn_p(x,p,Parameters_initial,Is_zero_p) for x in Age]
    
    ecart = 10
    
    Memory_density1 = [[Density[0:L]]]
    Memory_K1 = [[kernel_p(Time[0],x,0,List_Kmax,Parameters_kernel) for x in Age]]
    Memory_density2 = [[Density[L:2*L]]]
    Memory_K2 = [[kernel_p(Time[0],x,1,List_Kmax,Parameters_kernel) for x in Age]]
    Memory_density3 = [[Density[2*L:3*L]]]
    Memory_K3 = [[kernel_p(Time[0],x,2,List_Kmax,Parameters_kernel) for x in Age]]
    Memory_density4 = [[Density[3*L:4*L]]]
    Memory_K4 = [[kernel_p(Time[0],x,3,List_Kmax,Parameters_kernel) for x in Age]]
    Current_time = [Time[0]]
    compt = 0
    
    for i in range(len(Time)-1): # if we consider t = Time[-1], 
                         # we will calculate an approximation for a time bigger than T 
        t = Time[i]
        ts = Time[i+1]
        Density = step(t,Density,Age,delta_t,delta_x,kernel_p,List_Kmax,Parameters_kernel)
        if compt == ecart:
            t = Time[i+1]
            compt = 0
            Memory_density1 += [[Density[0:L]]]
            Memory_K1 += [[kernel_p(ts,x,0,List_Kmax,Parameters_kernel) for x in Age]]
            Memory_density2 += [[Density[L:2*L]]]
            Memory_K2 += [[kernel_p(ts,x,1,List_Kmax,Parameters_kernel) for x in Age]]
            Memory_density3 += [[Density[2*L:3*L]]]
            Memory_K3 += [[kernel_p(ts,x,2,List_Kmax,Parameters_kernel) for x in Age]]
            Memory_density4 += [[Density[3*L:4*L]]]
            Memory_K4 += [[kernel_p(ts,x,3,List_Kmax,Parameters_kernel) for x in Age]]
            Current_time += [ts]
        else :
            compt += 1
            
    fig, ax = plt.subplots(figsize=(12,10))
    plt.xlim(0,1.01*X)
    plt.ylim(0,1.5)
    plt.grid()
    plt.xlabel("Age")
    plt.ylabel("Value")
    plt.suptitle("Evolution of the density and the kernel over time t, \n T = %s , X = %s , delta_t = %s , delta_x = %s , I = %s , J = %s , \n Kmax1 = %s , sl1 = %s , sr1 = %s , xmin1 = %s , xmax1 = %s \n Kmax2 = %s , sl2 = %s , sr2 = %s , xmin2 = %s , xmax2 = %s \n Kmax3 = %s , sl3 = %s , sr3 = %s , xmin3 = %s , xmax3 = %s \n Kmax4 = %s , sl4 = %s , sr4 = %s , xmin4 = %s , xmax4 = %s , \n mu1 = %s , sigma1 = %s, mu2 = %s , sigma2 = %s , \n mu3 = %s , sigma3 = %s , mu4 = %s , sigma4 = %s"
              %(T,X,round(delta_t,5),round(delta_x,5),I,J,round(Kmax1,4),round(sl1,4),round(sr1,4),round(xmin1,4),round(xmax1,4),round(Kmax2,4),round(sl2,4),round(sr2,4),round(xmin2,4),round(xmax2,4),round(Kmax3,4),round(sl3,4),round(sr3,4),round(xmin3,4),round(xmax3,4),round(Kmax4,4),round(sl4,4),round(sr4,4),round(xmin4,4),round(xmax4,4),round(mu1,4),round(sigma1,4),round(mu2,4),round(sigma2,4),round(mu3,4),round(sigma3,4),round(mu4,4),round(sigma4,4)), fontsize=6)
    plt.legend(loc = 'upper left')

    kernelani1, = ax.plot([],[], color ='darkred' , linestyle = '--' , label = "Kernel from %s to %s"%(1,2))
    densityani1, = ax.plot([],[], color = 'red' , label = "Density phase %s"%(1))
    kernelani2, = ax.plot([],[], color = 'darkblue' , linestyle = '--' , label = "Kernel from %s to %s"%(2,3))
    densityani2, = ax.plot([],[], color = 'blue' ,label = "Density phase %s"%(2))
    kernelani3, = ax.plot([],[], color = 'darkgreen' , linestyle = '--' , label = "Kernel from %s to %s"%(3,4))
    densityani3, = ax.plot([],[], color = 'green' , label = "Density phase %s"%(3))
    kernelani4, = ax.plot([],[], color = 'darkmagenta' , linestyle = '--' , label = "Kernel from %s to %s"%(4,1))
    densityani4, = ax.plot([],[], color = 'magenta' , label = "Density phase %s"%(4))
    

    def init():
        kernelani1.set_data([], [])
        densityani1.set_data([], [])
        kernelani2.set_data([], [])
        densityani2.set_data([], [])
        kernelani3.set_data([], [])
        densityani3.set_data([], [])
        kernelani4.set_data([], [])
        densityani4.set_data([], [])
        plt.legend()
        return (kernelani1,densityani1,kernelani2,densityani2,kernelani3,densityani3,kernelani4,densityani4)

    def animate(i):
        t = Current_time[i]
        kernelani1.set_data(Age,Memory_K1[i])
        densityani1.set_data(Age,Memory_density1[i])
        kernelani2.set_data(Age,Memory_K2[i])
        densityani2.set_data(Age,Memory_density2[i])
        kernelani3.set_data(Age,Memory_K3[i])
        densityani3.set_data(Age,Memory_density3[i])
        kernelani4.set_data(Age,Memory_K4[i])
        densityani4.set_data(Age,Memory_density4[i])
        plt.title("Current time =%s"%(round(t,2)),fontsize = 10)
        plt.legend(loc = 'upper left')
        return (kernelani1,densityani1,kernelani2,densityani2,kernelani3,densityani3,kernelani4,densityani4)

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                             frames=len(Memory_K1), interval=100, blit=True,
                             repeat=True)
    
    if memo : # we save on demand
        anim.save(filename="%s.gif"%(name), dpi=80, fps=fps1)
    
    
    # Lignes de commandes permettant de créer un widget Javascript
    # interactif (représentation HTML par défaut des objets Animation)
    # à remplacer par la méthode plt.show() dans un éditeur python du
    # type spyder.
    # attention au paramétrage dans spyder :
    # menu outils> préférences> console IPython> Graphiques>
    # sortie graphique : automatique > OK
    # puis redémarrer spyder.
    # L'animation s'affichera dans une nouvelle fenêtre au lieu de
    # donner un graphique vierge dans le terminal)
    
    rc(animation)
    anim
    plt.close()
   
    
   
    
def mass(Vector,P,delta_x):
    Subvectors = separation(Vector, P)
    Mass_phase = []
    for p in range(P):
        sub_p = Subvectors[p]
        mass_p = (sub_p[0]+sub_p[-1])/2
        for j in range(1,len(sub_p)-1):
            mass_p += sub_p[j]
        mass_p = delta_x*mass_p
        Mass_phase +=[mass_p]
    mass_total = sum(Mass_phase)
    return(Mass_phase,mass_total)

def graph_mass(T,X,P,delta_x_wanted,initialn_p,kernel_p,List_Is_zero, List_Kmax,Parameters_kernel):
    plt.close()
    Kmax = max(List_Kmax)
    delta_t,delta_x,I,J,Time,Age = delta(T,X,delta_x_wanted,Kmax)
    
    
    Kmax1,Kmax2,Kmax3,Kmax4 = List_Kmax[0:4]
    mu1,sigma1 = Parameters_initial[0]
    mu2,sigma2 = Parameters_initial[1]
    mu3,sigma3 = Parameters_initial[2]
    mu4,sigma4 = Parameters_initial[3]
    sl1,sr1,xmin1,xmax1 = Parameters_kernel[0]
    sl2,sr2,xmin2,xmax2 = Parameters_kernel[1]
    sl3,sr3,xmin3,xmax3 = Parameters_kernel[2]
    sl4,sr4,xmin4,xmax4 = Parameters_kernel[3]
    
    
    Density = []
    Mass = []
    P = len(Parameters_initial)
    for p in range(P):
        Is_zero_p = List_Is_zero[p]
        Density += [initialn_p(x,p,Parameters_initial,Is_zero_p) for x in Age]
        
    Mass_phase,mass_total = mass(Density,P,delta_x)
    Mass += [mass_total]
        
    for i in range(len(Time)-1): # if we consider t = Time[-1], 
                         # we will calculate an approximation for a time bigger than T 
        Density = step(Time[i],Density,Age,delta_t,delta_x,kernel_p,List_Kmax,Parameters_kernel)
        Mass_phase,mass_total = mass(Density,P,delta_x)
        Mass += [mass_total]
        
    mass_min , mass_max = min(Mass) , max(Mass)
    m0 = Mass[0]
    pourc_max = 100*(mass_max/m0-1)
    pourc_min = 100*(mass_min/m0-1)
    plt.plot(Time,Mass)
    plt.suptitle("Evolution of the mass across time", fontsize = 32)
    plt.title("Pourcentage of error compared to the first mass between %s and %s \n T = %s , X = %s , delta_t = %s , delta_x = %s , I = %s , J = %s , \n Kmax1 = %s , sl1 = %s , sr1 = %s , xmin1 = %s , xmax1 = %s \n Kmax2 = %s , sl2 = %s , sr2 = %s , xmin2 = %s , xmax2 = %s \n Kmax3 = %s , sl3 = %s , sr3 = %s , xmin3 = %s , xmax3 = %s \n Kmax4 = %s , sl4 = %s , sr4 = %s , xmin4 = %s , xmax4 = %s , \n mu1 = %s , sigma1 = %s, mu2 = %s , sigma2 = %s , \n mu3 = %s , sigma3 = %s , mu4 = %s , sigma4 = %s"
              %(round(pourc_min,4),round(pourc_max,4),T,X,round(delta_t,5),round(delta_x,5),I,J,round(Kmax1,4),round(sl1,4),round(sr1,4),round(xmin1,4),round(xmax1,4),round(Kmax2,4),round(sl2,4),round(sr2,4),round(xmin2,4),round(xmax2,4),round(Kmax3,4),round(sl3,4),round(sr3,4),round(xmin3,4),round(xmax3,4),round(Kmax4,4),round(sl4,4),round(sr4,4),round(xmin4,4),round(xmax4,4),round(mu1,4),round(sigma1,4),round(mu2,4),round(sigma2,4),round(mu3,4),round(sigma3,4),round(mu4,4),round(sigma4,4)), fontsize=24)
    
    plt.legend()
    plt.show()
        
    
def animation_density_separated(T,X,P,delta_x_wanted,initialn_p,kernel_p,List_Is_zero,List_Kmax,Parameters_kernel,fps1,name,memo):
    """
    Animation of the first 4 densities across time (thus imposing P>=4)

    Parameters
    ----------
    T : float
        Maxmimum time.
    X : float
        Maximum age.
    P : int
        Number of phases.
    Delta_x_wanted : float
        Age step wanted.
    initialn_p : function
        Definition of the initial conditions (with P phases).
    kernel_p : function
        Definition of the kernel (with P phases).
    List_Is_zero : list of boolean
        If the initial density of phase p is null, then List_Is_zero[p] = True , else it is False
    List_Kmax : list of float
        List of maximum value of each kernel.
    Parameters_kernel : list of list of float
        List of parameters of each kernel.
    fps1 : int
        Number of FPS for the animation.
    name : char
        Name of the GIF.
    memo : bool
        True if we want to save the animation, False otherwise.

    Returns
    -------
    Animation of the first four phases

    """
    Kmax = max(List_Kmax)
    Kmax1,Kmax2,Kmax3,Kmax4 = List_Kmax[0:4]
    mu1,sigma1 = Parameters_initial[0]
    mu2,sigma2 = Parameters_initial[1]
    mu3,sigma3 = Parameters_initial[2]
    mu4,sigma4 = Parameters_initial[3]
    sl1,sr1,xmin1,xmax1 = Parameters_kernel[0]
    sl2,sr2,xmin2,xmax2 = Parameters_kernel[1]
    sl3,sr3,xmin3,xmax3 = Parameters_kernel[2]
    sl4,sr4,xmin4,xmax4 = Parameters_kernel[3]
    
    delta_t,delta_x,I,J,Time,Age = delta(T,X,delta_x_wanted,Kmax)
    L = J+1
    
    Density = []
    P = len(Parameters_initial)
    for p in range(P):
        Is_zero_p = List_Is_zero[p]
        Density += [initialn_p(x,p,Parameters_initial,Is_zero_p) for x in Age]
    
    ecart = 10
    
    Memory_density1 = [[Density[0:L]]]
    Memory_K1 = [[kernel_p(Time[0],x,0,List_Kmax,Parameters_kernel) for x in Age]]
    Memory_density2 = [[Density[L:2*L]]]
    Memory_K2 = [[kernel_p(Time[0],x,1,List_Kmax,Parameters_kernel) for x in Age]]
    Memory_density3 = [[Density[2*L:3*L]]]
    Memory_K3 = [[kernel_p(Time[0],x,2,List_Kmax,Parameters_kernel) for x in Age]]
    Memory_density4 = [[Density[3*L:4*L]]]
    Memory_K4 = [[kernel_p(Time[0],x,3,List_Kmax,Parameters_kernel) for x in Age]]
    Current_time = [Time[0]]
    compt = 0
    
    for i in range(len(Time)-1): # if we consider t = Time[-1], 
                         # we will calculate an approximation for a time bigger than T 
        t = Time[i]
        ts = Time[i+1]
        Density = step(t,Density,Age,delta_t,delta_x,kernel_p,List_Kmax,Parameters_kernel)
        if compt == ecart:
            compt = 0
            Memory_density1 += [[Density[0:L]]]
            Memory_K1 += [[kernel_p(ts,x,0,List_Kmax,Parameters_kernel) for x in Age]]
            Memory_density2 += [[Density[L:2*L]]]
            Memory_K2 += [[kernel_p(ts,x,1,List_Kmax,Parameters_kernel) for x in Age]]
            Memory_density3 += [[Density[2*L:3*L]]]
            Memory_K3 += [[kernel_p(ts,x,2,List_Kmax,Parameters_kernel) for x in Age]]
            Memory_density4 += [[Density[3*L:4*L]]]
            Memory_K4 += [[kernel_p(ts,x,3,List_Kmax,Parameters_kernel) for x in Age]]
            Current_time += [ts]
        else :
            compt += 1
            
    fig, axs = plt.subplots(2,2,figsize=(12,10))
    for absci in [0,1]:
        for ordo in [0,1]:
            axs[absci,ordo].set_ylim(0, 2)
            axs[absci,ordo].set_xlim(0, 1.01*X)
            axs[absci,ordo].set_xlabel("Age")
            axs[absci,ordo].set_ylabel("Value")
            axs[absci,ordo].grid()
            axs[absci,ordo].legend()
            
    plt.suptitle("Evolution of the density and the kernel over time t \n T = %s , X = %s , delta_t = %s , delta_x = %s , I = %s , J = %s , \n Kmax1 = %s , sl1 = %s , sr1 = %s , xmin1 = %s , xmax1 = %s \n Kmax2 = %s , sl2 = %s , sr2 = %s , xmin2 = %s , xmax2 = %s \n Kmax3 = %s , sl3 = %s , sr3 = %s , xmin3 = %s , xmax3 = %s \n Kmax4 = %s , sl4 = %s , sr4 = %s , xmin4 = %s , xmax4 = %s , \n mu1 = %s , sigma1 = %s, mu2 = %s , sigma2 = %s , \n mu3 = %s , sigma3 = %s , mu4 = %s , sigma4 = %s"
              %(T,X,round(delta_t,5),round(delta_x,5),I,J,round(Kmax1,4),round(sl1,4),round(sr1,4),round(xmin1,4),round(xmax1,4),round(Kmax2,4),round(sl2,4),round(sr2,4),round(xmin2,4),round(xmax2,4),round(Kmax3,4),round(sl3,4),round(sr3,4),round(xmin3,4),round(xmax3,4),round(Kmax4,4),round(sl4,4),round(sr4,4),round(xmin4,4),round(xmax4,4),round(mu1,4),round(sigma1,4),round(mu2,4),round(sigma2,4),round(mu3,4),round(sigma3,4),round(mu4,4),round(sigma4,4)), fontsize=6)
    

    kernelani1, = axs[0,0].plot([],[], color ='darkred' , linestyle = '--' , label = "Kernel from %s to %s"%(1,2))
    densityani1, = axs[0,0].plot([],[], color = 'red' , label = "Density phase %s"%(1))
    kernelani2, = axs[0,1].plot([],[], color = 'darkblue' , linestyle = '--' , label = "Kernel from %s to %s"%(2,3))
    densityani2, = axs[0,1].plot([],[], color = 'blue' ,label = "Density phase %s"%(2))
    kernelani3, = axs[1,0].plot([],[], color = 'darkgreen' , linestyle = '--' , label = "Kernel from %s to %s"%(3,4))
    densityani3, = axs[1,0].plot([],[], color = 'green' , label = "Density phase %s"%(3))
    kernelani4, = axs[1,1].plot([],[], color = 'darkmagenta' , linestyle = '--' , label = "Kernel from %s to %s"%(4,1))
    densityani4, = axs[1,1].plot([],[], color = 'magenta' , label = "Density phase %s"%(4))
    

    def init():
        kernelani1.set_data([], [])
        densityani1.set_data([], [])
        kernelani2.set_data([], [])
        densityani2.set_data([], [])
        kernelani3.set_data([], [])
        densityani3.set_data([], [])
        kernelani4.set_data([], [])
        densityani4.set_data([], [])
        for absci in [0,1]:
            for ordo in [0,1]:
                axs[absci,ordo].legend()
        return (kernelani1,densityani1,kernelani2,densityani2,kernelani3,densityani3,kernelani4,densityani4)

    def animate(i):
        t = Current_time[i]
        plt.title("Current time =%s"%(round(t,2)),fontsize = 8)
        kernelani1.set_data(Age,Memory_K1[i])
        densityani1.set_data(Age,Memory_density1[i])
        kernelani2.set_data(Age,Memory_K2[i])
        densityani2.set_data(Age,Memory_density2[i])
        kernelani3.set_data(Age,Memory_K3[i])
        densityani3.set_data(Age,Memory_density3[i])
        kernelani4.set_data(Age,Memory_K4[i])
        densityani4.set_data(Age,Memory_density4[i])
        
        for absci in [0,1]:
            for ordo in [0,1]:
                axs[absci,ordo].legend(loc = 'upper left')
        
                
        return (kernelani1,densityani1,kernelani2,densityani2,kernelani3,densityani3,kernelani4,densityani4)

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                             frames=len(Memory_K1), interval=100, blit=True,
                             repeat=True)
    
    if memo : # we save on demand
        anim.save(filename="%s.gif"%(name), dpi=80, fps=fps1)
    
    
    # Lignes de commandes permettant de créer un widget Javascript
    # interactif (représentation HTML par défaut des objets Animation)
    # à remplacer par la méthode plt.show() dans un éditeur python du
    # type spyder.
    # attention au paramétrage dans spyder :
    # menu outils> préférences> console IPython> Graphiques>
    # sortie graphique : automatique > OK
    # puis redémarrer spyder.
    # L'animation s'affichera dans une nouvelle fenêtre au lieu de
    # donner un graphique vierge dans le terminal)
    
    rc(animation)
    plt.close()
