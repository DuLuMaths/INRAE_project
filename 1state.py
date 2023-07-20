""" Finite difference method for 1 phase """

if __name__== "__main__":
    
    # Parameters of our scheme
    # In general, we will set at most
    T = 150.0 # final time
    X = 25.0 # maximum age such that we don't miss information in true differential system
    Kmax = 5.0
    delta_x = 1/10
    delta_t = min(1/3*delta_x,delta_x/(1+Kmax*delta_x))
    I = int(T/delta_t)
    J = int(X/delta_x)
    mu = 3.0
    sigma = 0.5
    s1 = 5.0
    s2 = 5.0
    xmin = 15.0
    xmax = 24.0
    eps_kernel = 10**(-3)
    eps_density = 10**(-3)
    fps1 = 8 # Number of frame per second for the animations
    name = "Test" # Name for the animation
    memo = True # If we save the animation
    


# Modules

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import animation , rc
import seaborn as sns
import pandas as pd



def delta(T,X,delta_x,Kmax):
    """
    CFL conditions :
    We want Delta_t / Delta_x = a < 1/2 and Delta_t < Delta_x /(1 + Kmax Delta_x)
    thus we have to choose Delta_x < (1/Kmax)(1/a - 1)

    Parametersmax,maxratio
    ----------
    T : float
        Final time of the scheme.
    X : float
        Maximum age of the scheme.
    delta_x : float
        Age step.
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
    J : intcalcul_x_obs(Kernel,Age,Kmax,s1,s2,xmin,xmax,eps)
        J+1 number of element in Age.
    Time : list of float
            Discretisation of time.
    Age : list of float
            Discretisation of age.
    """
    
    delta_t = min(1/3*delta_x,delta_x/(1+Kmax*delta_x))
    I = int(T/delta_t)
    J = int(X/delta_x)
    # the choice of I and J impose that the CFL conditions are still true
    Time = np.array([i*delta_t for i in range(I+1)]) # discretisation of our time
    Age = np.array([j*delta_x for j in range(J+1)]) # discretisation of our age
    
    return(delta_t,delta_x,I,J,Time,Age)


if __name__== "__main__":
    delta_t,delta_x,I,J,Time,Age = delta(T,X,delta_x,Kmax)


def initial_n(x,mu,sigma): # initial condition on the density n
    result = np.exp(-(x-mu)**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)
    return(result)   



def Kernel(t,x,Kmax,s1,s2,xmin,xmax): # definition of the kernel K
    result = (np.tanh(s1*(x-(xmin+1.5*np.cos(t)))) + np.tanh(s2*(xmax-x)))*Kmax/2
    return(result)
    


def matrix_transition(t, Age, Kernel, Kmax, delta_t, delta_x,s1,s2,xmin,xmax,J):
    """
    Creation of the matrix of transition between two time steps of our scheme

    Parameters
    ----------
    t : float
        Time at which we make the calculations.
    Age : list of float
        All ages we consider for our discretisation.
    Kernel : function
        Kernel of transition taking four inputs (t,x,X_older,Kmax) and return a float.
    Kmax : float
        Upper bound of the kernel.
    delta_t : float
        Time step.
    delta_x : float
        Age step.
    s1,s2,xmin,xmax : float
        Parameters for the Kernel.
    J : int
        The result matrix is of size (J+1,J+1)

    Returns
    -------
    mtx : array of float (CSR format)
        Matrix of the transition.

    """   
    ratio = delta_t/delta_x
    K = [Kernel(t,x,Kmax,s1,s2,xmin,xmax) for x in Age]
    
    
    Row = []
    Col = []
    Data = []
    
    # First row
    Row += [0 for j in range(J+1)]
    Col += [j for j in range(J+1)]
    Data += [1-2*ratio] # First column
    Data += [2*delta_t*K[j] for j in range(1,J-1)] # Middle columns
    Data += [2*delta_t*K[J-1]+ratio] # J-1 column
    Data += [delta_t*K[J] + ratio]
    
    # Other rows
    #     Subdiagonal
    Row += [i for i in range(1,J+1)]
    Col += [j for j in range(J)]
    Data += [ratio for j in range(J)]
    #     Diagonal
    Row += [i for i in range(1,J+1)]
    Col += [j for j in range(1,J+1)]
    Data += [1-ratio-delta_t*K[j] for j in range(1,J+1)]
    
    
    row = np.array(Row)
    col = np.array(Col)
    data = np.array(Data)
    mtx = sp.sparse.coo_matrix((data, (row, col)), shape=(J+1, J+1))
    return(mtx)


def heatmap(t,Age, Kernel, Kmax, delta_t, delta_x,s1,s2,xmin,xmax,J):
    """
    Heatmap representation of the matrix of transition at time 
    """
    plt.close()
    mtx = matrix_transition(t, Age, Kernel, Kmax, delta_t, delta_x,s1,s2,xmin,xmax,J)
    Mtx = mtx.todense()
    plt.gcf().suptitle("Matrix of transition at time t=%s"%(t))
    heatmap = sns.heatmap(Mtx)
    return(heatmap)


def step(t, Age, Density_old, Kernel, Kmax, delta_t, delta_x,s1,s2,xmin,xmax,J):
    """
    Given an approximation of our density at a time t,
    returns an approximation at the next time step

    Parameters
    ----------
    t : float
        Time at which we make the calculations.
    Age : list of float
        All ages we consider for our discretisation.
    Density_old : array of float
        Approximation of our density at time t.
    Kernel : function
        Kernel of transition taking four float inputs (t,x,X_older,Kmax) and return a float.
    Kmax : float
        Upper bound of the kernel.
    delta_t : float
        Time step.
    delta_x : float
        Age step.
    s1,s2,xmin,xmax : float
        Parameters for the Kernel.
    J : int
        (J+1,J+1) size of the matrix of transition.

    Returns
    -------
    Density_new : array of float
        Approximation of the density at the next time step

    """
    M = matrix_transition(t, Age, Kernel, Kmax, delta_t, delta_x,s1,s2,xmin,xmax,J)
    Density_new = M@Density_old
    return(Density_new)



def graph(T, X, delta_x, initial_n, Kernel, Kmax,s1,s2,xmin,xmax,mu,sigma):
    """
    Show the approximated density at the final time

    Parameters
    ----------
    T : float
        Maximum time.
    X : float
        Maximum age.
    delta_x : float
        Age step.
    initial_n : function
        Description of our initial density. Takes two float inputs (x,X_older) and return a float.
    Kernel : function
        Kernel of transition taking four float inputs (t,x,X_older,Kmax) and return a float.
    Kmax : float
        Upper bound of the kernel.
    s1,s2,xmin,xmax,mu,sigma : float
        Parameters for our initial condition and for the Kernel.

    Returns
    -------
    Graph of the approximated density at the final time.

    """
    delta_t,delta_x,I,J,Time,Age = delta(T,X,delta_x,Kmax)
    Density = np.array([initial_n(x,mu,sigma) for x in Age]) 
    for t in Time[0:-1]: # if we consider t = Time[-1], 
                         # we will calculate an approximation for a time bigger than T 
        Density = step(t, Age, Density, Kernel, Kmax, delta_t, delta_x,s1,s2,xmin,xmax,J)
        
    plt.plot(Age,Density)
    plt.xlim(0,X)
    plt.ylim(0,5)
    plt.suptitle("Approximated density at time T={}".format(T))
    plt.title("T=%s , X=%s , delta_x=%s , delta_t=%s , I= %s , J=%s ,\n Kmax=%s , xmin=%s , xmax=%s , s1=%s , s2=%s ,\n mu=%s , sigma=%s"%(T,X, round(delta_x,5), round(delta_t,5),I,J, Kmax, xmin,xmax,s1,s2,mu,sigma),fontsize = 20)
    plt.xlabel("Age x")
    plt.ylabel("Density n(T,x)")
    plt.legend()
    plt.show()
    plt.close()
    
    

def mass(vect,delta_x):
    """
    Trapezoidal method to approximate the mass (=area) defined by a vector
    which is an approximation of a function.

    Parameters
    ----------
    vect : array of float
        Vector that approximate a function.
    delta_x : float
        Uniform age step.

    Returns
    -------
    area : float
        Approximation of the area

    """
    L = np.size(vect)
    area = 0
    for l in range(1,L-1):
        area += vect[l]
    area = (delta_x/2)*(vect[0] + 2*area + vect[-1])
    return(area)
    


def mass_conservation(T, X, delta_x, initial_n, Kernel, Kmax,s1,s2,xmin,xmax,mu,sigma):
    """
    Verification of the conservation of mass over time

    Parameters
    ----------
    T : float
        Maximum time.
    X : float
        Maximum age.
    delta_x : float
        Age step.
    initial_n : function
        Description of our initial density. Takes two float inputs (x,X_older) and return a float.
    Kernel : function
        Kernel of transition taking four float inputs (t,x,X_older,Kmax) and return a float.
    Kmax : float
        Upper bound of the kernel.
    s1,s2,xmin,xmax,mu,sigma : float
        Parameters for our initial condition and for the Kernel.


    Returns
    -------
    Graph of the approximated mass over time

    """
    plt.close()
    
    delta_t,delta_x,I,J,Time,Age = delta(T,X,delta_x,Kmax)
    Density = np.array([initial_n(x,mu,sigma) for x in Age]) 
    D = [Density]
    Mass = [mass(Density,delta_x)]
    for t in Time[0:-1]: # if we consider t = Time[-1], 
                         # we will calculate an approximation for a time bigger than T 
        Density = step(t, Age, Density, Kernel, Kmax, delta_t, delta_x,s1,s2,xmin,xmax,J)
        D += [Density]
        Mass += [mass(Density,delta_x)]
        
    plt.plot(Time,Mass)
    plt.xlim(0,T)
    plt.ylim(0.99,1.01)
    plt.suptitle("Approximated mass over time ",fontsize = 28)
    plt.title("T=%s , X=%s , delta_x=%s , delta_t=%s , I= %s , J=%s ,\n Kmax=%s , xmin=%s , xmax=%s , s1=%s , s2=%s ,\n mu=%s , sigma=%s"%(T,X, round(delta_x,5), round(delta_t,5),I,J, Kmax, xmin,xmax,s1,s2,mu,sigma),fontsize = 20)
    plt.xlabel("Time t")
    plt.ylabel("Mass M(t)")
    plt.legend()
    plt.show()
    
    
    

def animation_density(T, X, delta_x, initial_n, Kernel, Kmax,s1,s2,xmin,xmax,mu,sigma,fps1,name,memo):
    """
    Create an animation of the evolution of density over time

    Parameters
    ----------
    T : float
        Maximum time.
    X : float
        Maximum age.
    initial_n : function
        Description of our initial density. Takes two float inputs (x,X_older) and return a float.
    Kernel : function
        Kernel of transition taking four float inputs (t,x,X_older,Kmax) and return a float.
    Kmax : float
        Upper bound of the kernel.
    s1,s2,xmin,xmax,mu,sigma : float
        Parameters for our initial condition and for the Kernel.
    fps1 : int
        Number of FPS for the animation.
    name : character
        Name for the file.
    memo : boolean
        True if we want to save the GIF.


    Returns
    -------
    Graph of the approximated density at the final time.

    """
    delta_t,delta_x,I,J,Time,Age = delta(T,X,delta_x,Kmax)
    Density = np.array([initial_n(x,mu,sigma) for x in Age]) 
    
    ecart = 10
    Liste = [Density]
    K = [[Kernel(0,x,Kmax,s1,s2,xmin,xmax) for x in Age]]
    Current_time = [Time[0]]
    compt = 0
    for i in range(len(Time)-1): # if we consider t = Time[-1], 
                         # we will calculate an approximation for a time bigger than T 
        t = Time[i]
        ts = Time[i+1]
        Density = step(t, Age, Density, Kernel, Kmax, delta_t, delta_x,s1,s2,xmin,xmax,J)
        if compt == ecart:
            compt = 0
            Liste += [Density]
            K += [[Kernel(t,x,Kmax,s1,s2,xmin,xmax) for x in Age]]
            Current_time += [ts]
        else :
            compt += 1
        
    
    
    fig, ax = plt.subplots(figsize=(12,10))
    plt.xlim(0,20)
    plt.ylim(0,2.5)
    plt.grid()
    plt.xlabel("Age")
    plt.ylabel("Value")
    plt.suptitle("Evolution of the density and the kernel over time, \n T = %s , X = %s , delta_t = %s , delta_x = %s , I = %s , J = %s  ,\n Kmax = %s, s1 = %s , s2 = %s , xmin = %s , xmax = %s, mu = %s , sigma = %s "
              %(T,X,round(delta_t,5),round(delta_x,5),I,J,Kmax,s1,s2,xmin,xmax,mu,sigma), fontsize=18)
    plt.legend()


    line, = ax.plot([],[], 'r--' , label = "Kernel")
    point, = ax.plot([],[], 'b', label = "Density")

    def init():
        line.set_data([], [])
        point.set_data([], [])
        plt.legend()
        return (line,point)

    def animate(i):
        t = Current_time[i]
        x = Age
        y = Liste[i]
        z = K[i]
        line.set_data(x,z)
        point.set_data(x,y)
        plt.title("Current time = %s"%(round(t,2)),fontsize = 16)
        plt.legend()
        return (line,point)

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                             frames=len(Liste), interval=100, blit=True,
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
    

def calcul_mean(Weight,Value):
    L = np.size(Weight)
    Sum = 0
    s = 0
    for l in range(L):
        s += Weight[l]
        Sum += Weight[l]*Value[l]
    S = Sum/s
    return(S)


def graph_mean(T, X, delta_x, initial_n, Kernel, Kmax,s1,s2,xmin,xmax,mu,sigma):
    """
    Graph of the mean over time

    Parameters
    ----------
    T : float
        Maximum time.
    X : float
        Maximum age.
    delta_x : float
        Age step.
    initial_n : function
        Description of our initial density. Takes two float inputs (x,X_older) and return a float.
    Kernel : function
        Kernel of transition taking four float inputs (t,x,X_older,Kmax) and return a float.
    Kmax : float
        Upper bound of the kernel.
    s1,s2,xmin,xmax,mu,sigma : float
        Parameters for our initial condition and for the Kernel.


    Returns
    -------
    Graph of the approximated mean over time

    """
    plt.close()
    delta_t,delta_x,I,J,Time,Age = delta(T,X,delta_x,Kmax)
    Density = np.array([initial_n(x,mu,sigma) for x in Age]) 
    Mean = [calcul_mean(Density,Age)]
    for t in Time[0:-1]: # if we consider t = Time[-1], 
                         # we will calculate an approximation for a time bigger than T 
        Density = step(t, Age, Density, Kernel, Kmax, delta_t, delta_x,s1,s2,xmin,xmax,J)
        Mean += [calcul_mean(Density,Age)]
        
    plt.plot(Time,Mean)
    plt.xlim(0,T)
    plt.suptitle("Approximated mean over time", fontsize = 28)
    plt.title("T=%s , X=%s , delta_x=%s , delta_t=%s , I= %s , J=%s ,\n Kmax=%s , xmin=%s , xmax=%s , s1=%s , s2=%s ,\n mu=%s , sigma=%s"%(T,X, round(delta_x,5), round(delta_t,5),I,J, Kmax, xmin,xmax,s1,s2,mu,sigma),fontsize = 20)
    plt.xlabel("Time")
    plt.ylabel("Mean")
    plt.legend()
    plt.show()
    
    
# To see if the slope is almost constant when the support of the density
# is disjoint from the support of the Kernel :
    
def derive_approx(Vector,Absci):
    Deriv = []
    for i in range(np.size(Vector)-1):
        Deriv += [(Vector[i+1]-Vector[i])/(Absci[i+1]-Absci[i])]
    return(Deriv)
    
def graph_der_mean(T, X, delta_x, initial_n, Kernel, Kmax,s1,s2,xmin,xmax,mu,sigma):
    """
    Graph of the mean over time

    Parameters
    ----------
    T : float
        Maximum time.
    X : float
        Maximum age.
    delta_x : float
        Age step.
    initial_n : function
        Description of our initial density. Takes two float inputs (x,X_older) and return a float.
    Kernel : function
        Kernel of transition taking four float inputs (t,x,X_older,Kmax) and return a float.
    Kmax : float
        Upper bound of the kernel.
    s1,s2,xmin,xmax,mu,sigma : float
        Parameters for our initial condition and for the Kernel.


    Returns
    -------
    Graph of the approximated mean over time

    """
    plt.close()
    delta_t,delta_x,I,J,Time,Age = delta(T,X,delta_x,Kmax)
    Density = np.array([initial_n(x,mu,sigma) for x in Age]) 
    Mean = [calcul_mean(Density,Age)]
    for t in Time[0:-1]: # if we consider t = Time[-1], 
                         # we will calculate an approximation for a time bigger than T 
        Density = step(t, Age, Density, Kernel, Kmax, delta_t, delta_x,s1,s2,xmin,xmax,J)
        Mean += [calcul_mean(Density,Age)]
        
    Deriv = derive_approx(Mean,Time)
        
    plt.plot(Time[0:-1],Deriv)
    plt.xlim(0,T)
    plt.suptitle("Approximated derivative of the mean over time", fontsize = 28)
    plt.title("T=%s , X=%s , delta_x=%s , delta_t=%s , I= %s , J=%s ,\n Kmax=%s , xmin=%s , xmax=%s , s1=%s , s2=%s ,\n mu=%s , sigma=%s"%(T,X, round(delta_x,5), round(delta_t,5),I,J, Kmax, xmin,xmax,s1,s2,mu,sigma),fontsize = 20)
    plt.xlabel("Time")
    plt.ylabel("Derivative of the mean")
    plt.legend()
    plt.show()
    
    
def calcul_variance(Weight,Value,mean):
    L = np.size(Weight)
    Sum = 0
    s = 0
    for l in range(L):
        s += Weight[l]
        Sum += Weight[l]*(Value[l]-mean)**2
    S = Sum/s
    return(S)

def graph_sd(T, X, delta_x, initial_n, Kernel, Kmax,s1,s2,xmin,xmax,mu,sigma):
    """
    Graph of the standard deviation over time

    Parameters
    ----------
    T : float
        Maximum time.
    X : float
        Maximum age.
    delta_x : float
        Age step.
    initial_n : function
        Description of our initial density. Takes two float inputs (x,X_older) and return a float.
    Kernel : function
        Kernel of transition taking four float inputs (t,x,X_older,Kmax) and return a float.
    Kmax : float
        Upper bound of the kernel.
    s1,s2,xmin,xmax,mu,sigma : float
        Parameters for our initial condition and for the Kernel.


    Returns
    -------
    Graph of the approximated standard deviation over time

    """
    plt.close()
    delta_t,delta_x,I,J,Time,Age = delta(T,X,delta_x,Kmax)
    Density = np.array([initial_n(x,mu,sigma) for x in Age]) 
    mean = calcul_mean(Density,Age)
    Sd = [np.sqrt(calcul_variance(Density, Age, mean))]
    for t in Time[0:-1]: # if we consider t = Time[-1], 
                         # we will calculate an approximation for a time bigger than T 
        Density = step(t, Age, Density, Kernel, Kmax, delta_t, delta_x,s1,s2,xmin,xmax,J)
        mean = calcul_mean(Density,Age)
        Sd += [np.sqrt(calcul_variance(Density, Age, mean))]
        
    plt.plot(Time,Sd)
    plt.xlim(0,T)
    plt.suptitle("Approximated standard deviation over time ",fontsize = 28)
    plt.title("T=%s , X=%s , delta_x=%s , delta_t=%s , I= %s , J=%s ,\n Kmax=%s , xmin=%s , xmax=%s , s1=%s , s2=%s ,\n mu=%s , sigma=%s"%(T,X, round(delta_x,5), round(delta_t,5),I,J, Kmax, xmin,xmax,s1,s2,mu,sigma),fontsize = 20)
    plt.xlabel("Time")
    plt.ylabel("Standard deviation")
    plt.legend()
    plt.show()
    

    
def animation_mean_sd(T, X, delta_x, initial_n, Kernel, Kmax,s1,s2,xmin,xmax,mu,sigma,fps1,name,memo):
    """
    Create an animation of the evolution of density over time, with the position of the estimated mean

    Parameters
    ----------
    T : float
        Maximum time.
    X : float
        Maximum age.
    initial_n : function
        Description of our initial density. Takes two float inputs (x,X_older) and return a float.
    Kernel : function
        Kernel of transition taking four float inputs (t,x,X_older,Kmax) and return a float.
    Kmax : float
        Upper bound of the kernel.
    s1,s2,xmin,xmax,mu,sigma : float
        Parameters for our initial condition and for the Kernel.
    fps1 : int
        Number of FPS for the animation.
    name : character
        Name for the file.
    memo : boolean
        True if we want to save the GIF.the


    Returns
    -------
    Graph of the approximated density at the final time.

    """
    delta_t,delta_x,I,J,Time,Age = delta(T,X,delta_x,Kmax)
    Density = np.array([initial_n(x,mu,sigma) for x in Age]) 
    
    ecart = 10
    Liste = [Density]
    K = [[Kernel(0,x,Kmax,s1,s2,xmin,xmax) for x in Age]]
    mean = calcul_mean(Density,Age)
    Mean = [mean]
    Sd = [np.sqrt(calcul_variance(Density, Age, mean))]
    Current_time = [Time[0]]
    compt = 0
    for i in range(len(Time)-1): # if we consider t = Time[-1], 
                         # we will calculate an approximation for a time bigger than T 
        t = Time[i]
        ts = Time[i+1]
        Density = step(t, Age, Density, Kernel, Kmax, delta_t, delta_x,s1,s2,xmin,xmax,J)
        if compt == ecart:
            compt = 0
            Liste += [Density]
            K += [[Kernel(ts,x,Kmax,s1,s2,xmin,xmax) for x in Age]]
            mean = calcul_mean(Density,Age)
            Mean += [mean]
            Sd += [np.sqrt(calcul_variance(Density, Age, mean))]
            Current_time += [ts]
        else :
            compt += 1
        
    
    
    fig, ax = plt.subplots(figsize=(12,10))
    plt.xlim(0,20)
    plt.ylim(0,2.5)
    plt.grid()
    plt.xlabel("Age")
    plt.ylabel("Value")
    plt.suptitle("Evolution of the density and the kernel over time t, \n T = %s , X = %s , delta_t = %s , delta_x = %s , I = %s , J = %s  ,\n Kmax = %s, s1 = %s , s2 = %s , xmin = %s , xmax = %s, mu = %s , sigma = %s"
              %(T,X,round(delta_t,5),round(delta_x,5),I,J,Kmax,s1,s2,xmin,xmax,mu,sigma), fontsize=18)
    plt.legend()


    line, = ax.plot([],[], 'r--' , label = "Kernel")
    point, = ax.plot([],[], 'b', label = "Density")
    verti_mean, = ax.plot([],[], 'g--' , label = "Mean")
    verti_sd_left, = ax.plot([],[], color = "dimgrey", linestyle = '--' , label = "+1sigma")
    verti_sd_right, = ax.plot([],[], color = "dimgrey", linestyle = '--' , label = "-1sigma")

    def init():
        line.set_data([], [])
        point.set_data([], [])
        verti_mean.set_data([], [])
        verti_sd_left.set_data([], [])
        verti_sd_right.set_data([], [])
        plt.legend()
        return (line,point,verti_mean,verti_sd_left,verti_sd_right)

    def animate(i):
        t = Current_time[i]
        x = Age
        y = Liste[i]
        z = K[i]
        m = Mean[i]
        sig = Sd[i]
        line.set_data(x,z)
        point.set_data(x,y)
        verti_mean.set_data([m,m],[0,Kmax])
        verti_sd_left.set_data([m-sig,m-sig],[0,Kmax])
        verti_sd_right.set_data([m+sig,m+sig],[0,Kmax])
        plt.title("Current time = %s"%(round(t,2)),fontsize = 16)
        plt.legend()
        return (line,point,verti_mean,verti_sd_left,verti_sd_right)

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                             frames=len(Liste), interval=100, blit=True,
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
    
# The mean and standard deviation we look at are calculated based on every point where we have information
# Thus we consider the influence of the region defined by the support of the kernel
# This introduced modifications of the mean and sd we want to look at
# The mean and sd we want are coming from the distribution not significantly influence by the kernel at time t
# Has a first try, we introduce a window of observation [0,x_obs] with x_obs outside the support of the kernel



# We will consider the kernel independant of time
# Just to see what happen, we will trace out the mean and sd from the distribution inside the window
# even if there is a lot of cells outside the window of observation

def calcul_x_obs(Kernel,t,Age,Kmax,s1,s2,xmin,xmax,eps):
    x_obs = Age[0]
    index_obs = 0
    for x in Age[1:-1]:
        if Kernel(t,x,Kmax,s1,s2,xmin,xmax) < eps:
            index_obs += 1
            x_obs = x
        else :
            return(x_obs,index_obs)
    return(x_obs,index_obs)
        

def animationtestwindow_mean_sd(T, X, delta_x, initial_n, Kernel, Kmax,s1,s2,xmin,xmax,mu,sigma,eps,fps1,name,memo):
    """
    Create an animation of the evolution of density over time, with the position of the estimated mean

    Parameters
    ----------
    T : float
        Maximum time.
    X : float
        Maximum age.
    initial_n : function
        Description of our initial density. Takes two float inputs (x,X_older) and return a float.
    Kernel : function
        Kernel of transition taking four float inputs (t,x,X_older,Kmax) and return a float.
    Kmax : float
        Upper bound of the kernel.
    s1,s2,xmin,xmax,mu,sigma : float
        Parameters for our initial condition and for the Kernel.
    fps1 : int
        Number of FPS for the animation.
    name : character
        Name for the file.
    memo : boolean
        True if we want to save the GIF.


    Returns
    -------
    Graph of the approximated density at the final time.

    """
    delta_t,delta_x,I,J,Time,Age = delta(T,X,delta_x,Kmax)
    x_obs,index_obs = calcul_x_obs(Kernel,Time[0],Age,Kmax,s1,s2,xmin,xmax,eps)
    Density = np.array([initial_n(x,mu,sigma) for x in Age]) 
    
    ecart = 10
    Liste = [Density]
    K = [[Kernel(0,x,Kmax,s1,s2,xmin,xmax) for x in Age]]
    mean = calcul_mean(Density[0:index_obs+1],Age[0:index_obs+1])
    Mean = [mean]
    Sd = [np.sqrt(calcul_variance(Density[0:index_obs+1], Age[0:index_obs+1], mean))]
    Obs = [x_obs]
    Current_time = [Time[0]]
    compt = 0
    for i in range(len(Time)-1): # if we consider t = Time[-1], 
                         # we will calculate an approximation for a time bigger than T 
        t = Time[i]
        ts = Time[i+1]
        Density = step(t, Age, Density, Kernel, Kmax, delta_t, delta_x,s1,s2,xmin,xmax,J)
        x_obs,index_obs = calcul_x_obs(Kernel,ts,Age,Kmax,s1,s2,xmin,xmax,eps)
        if compt == ecart:
            compt = 0
            Liste += [Density]
            K += [[Kernel(ts,x,Kmax,s1,s2,xmin,xmax) for x in Age]]
            mean = calcul_mean(Density[0:index_obs+1],Age[0:index_obs+1])
            Mean += [mean]
            Sd += [np.sqrt(calcul_variance(Density[0:index_obs+1], Age[0:index_obs+1], mean))]
            Obs += [x_obs]
            Current_time += [ts]
        else :
            compt += 1
        
    
    
    fig, ax = plt.subplots(figsize=(12,10))
    plt.xlim(0,20)
    plt.ylim(0,2.5)
    plt.grid()
    plt.xlabel("Age")
    plt.ylabel("Value")
    plt.suptitle("Evolution of the density and the kernel over time t, \n T = %s , X = %s , delta_t = %s , delta_x = %s , I = %s , J = %s  , Kmax = %s, s1 = %s , s2 = %s , xmin = %s , xmax = %s, mu = %s , sigma = %s"
              %(T,X,round(delta_t,5),round(delta_x,5),I,J,Kmax,s1,s2,xmin,xmax,mu,sigma), fontsize=18)
    plt.legend()


    line, = ax.plot([],[], 'r--' , label = "Kernel")
    point, = ax.plot([],[], 'b', label = "Density")
    verti_mean, = ax.plot([],[], 'g--' , label = "Mean")
    verti_sd_left, = ax.plot([],[], color = "dimgrey", linestyle = '--' , label = "+1sigma")
    verti_sd_right, = ax.plot([],[], color = "dimgrey", linestyle = '--' , label = "-1sigma")
    verti_obs, = ax.plot([],[], color = "black", linestyle = '-' , label = "x_obs")

    def init():
        line.set_data([], [])
        point.set_data([], [])
        verti_mean.set_data([], [])
        verti_sd_left.set_data([], [])
        verti_sd_right.set_data([], [])
        verti_obs.set_data([], [])
        plt.legend()
        return (line,point,verti_mean,verti_sd_left,verti_sd_right,verti_obs)

    def animate(i):
        t = Current_time[i]
        x = Age
        y = Liste[i]
        z = K[i]
        m = Mean[i]
        sig = Sd[i]
        x_obs = Obs[i]
        line.set_data(x,z)
        point.set_data(x,y)
        verti_mean.set_data([m,m],[0,Kmax])
        verti_sd_left.set_data([m-sig,m-sig],[0,Kmax])
        verti_sd_right.set_data([m+sig,m+sig],[0,Kmax])
        verti_obs.set_data([x_obs,x_obs],[0,Kmax])
        plt.title("Current time =%s"%(round(t,2)),fontsize=16)
        plt.legend()
        return (line,point,verti_mean,verti_sd_left,verti_sd_right,verti_obs)

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                             frames=len(Liste), interval=100, blit=True,
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
    
def graphtestwindow_mean_sd(T, X, delta_x, initial_n, Kernel, Kmax,s1,s2,xmin,xmax,mu,sigma,eps):
    """
    Graph of the mean over time

    Parameters
    ----------
    T : float
        Maximum time.
    X : float
        Maximum age.
    delta_x : float
        Age step.
    initial_n : function
        Description of our initial density. Takes two float inputs (x,X_older) and return a float.
    Kernel : function
        Kernel of transition taking four float inputs (t,x,X_older,Kmax) and return a float.
    Kmax : float
        Upper bound of the kernel.
    s1,s2,xmin,xmax,mu,sigma : float
        Parameters for our initial condition and for the Kernel.


    Returns
    -------
    Graph of the approximated mean over time

    """
    plt.close()
    delta_t,delta_x,I,J,Time,Age = delta(T,X,delta_x,Kmax)
    x_obs,index_obs = calcul_x_obs(Kernel,Time[0],Age,Kmax,s1,s2,xmin,xmax,eps)
    
    Density = np.array([initial_n(x,mu,sigma) for x in Age]) 
    mean = calcul_mean(Density[0:index_obs+1],Age[0:index_obs+1])
    Mean = [mean]
    Sd = [np.sqrt(calcul_variance(Density[0:index_obs+1], Age[0:index_obs+1], mean))]
    for i in range(len(Time)-1): # if we consider t = Time[-1], 
                         # we will calculate an approximation for a time bigger than T 
        t = Time[i]
        ts = Time[i+1]
        Density = step(t, Age, Density, Kernel, Kmax, delta_t, delta_x,s1,s2,xmin,xmax,J)
        x_obs,index_obs = calcul_x_obs(Kernel,ts,Age,Kmax,s1,s2,xmin,xmax,eps)
        mean = calcul_mean(Density[0:index_obs+1],Age[0:index_obs+1])
        Mean += [mean]
        Sd += [np.sqrt(calcul_variance(Density[0:index_obs+1], Age[0:index_obs+1],mean))]
        
    plt.plot(Time,Mean,label="Mean")
    plt.plot(Time,Sd,label="Standard deviation")
    plt.xlim(0,T)
    plt.suptitle("Approximated mean and standard deviation over time ",fontsize = 28)
    plt.title("T=%s , X=%s , delta_x=%s , delta_t=%s , I= %s , J=%s ,\n Kmax=%s , xmin=%s , xmax=%s , s1=%s , s2=%s ,\n mu=%s , sigma=%s"%(T,X, round(delta_x,5), round(delta_t,5),I,J, Kmax, xmin,xmax,s1,s2,mu,sigma),fontsize = 20)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.show()
    
    
# Now we want to only have the mean and sd when almost all cells are in the window of observation
# and 

def calcul_x_end_obs(Kernel,t,Age,Kmax,s1,s2,xmin,xmax,eps):
    RevAge = list(reversed(Age))
    x_end_obs,index = calcul_x_obs(Kernel,t,RevAge,Kmax,s1,s2,xmin,xmax,eps)
    index_end_obs = np.size(Age)-1-index
    return(x_end_obs,index_end_obs)


def animationwindow_mean_sd(T, X, delta_x, initial_n, Kernel, Kmax,s1,s2,xmin,xmax,mu,sigma,eps,fps1,name,memo):
    """
    Create an animation of the evolution of density over time, with the position of the estimated mean

    Parameters
    ----------
    T : float
        Maximum time.
    X : float
        Maximum age.
    initial_n : function
        Description of our initial density. Takes two float inputs (x,X_older) and return a float.
    Kernel : function
        Kernel of transition taking four float inputs (t,x,X_older,Kmax) and return a float.
    Kmax : float
        Upper bound of the kernel.
    s1,s2,xmin,xmax,mu,sigma : float
        Parameters for our initial condition and for the Kernel.
    fps1 : int
        Number of FPS for the animation.
    name : character
        Name for the file.
    memo : boolean
        True if we want to save the GIF.


    Returns
    -------
    Graph of the approximated density at the final time.

    """
    delta_t,delta_x,I,J,Time,Age = delta(T,X,delta_x,Kmax)
    x_obs,index_obs = calcul_x_obs(Kernel,Time[0],Age,Kmax,s1,s2,xmin,xmax,eps)
    Density = np.array([initial_n(x,mu,sigma) for x in Age]) 
    
    ecart = 10
    Liste = [Density]
    K = [[Kernel(0,x,Kmax,s1,s2,xmin,xmax) for x in Age]]
    mean = calcul_mean(Density[0:index_obs+1],Age[0:index_obs+1])
    Mean = [mean]
    Sd = [np.sqrt(calcul_variance(Density[0:index_obs+1], Age[0:index_obs+1], mean))]
    Obs = [x_obs]
    Current_time = [Time[0]]
    compt = 0
    for i in range(len(Time)-1): # if we consider t = Time[-1], 
                         # we will calculate an approximation for a time bigger than T 
        t = Time[i]
        ts = Time[i+1]
        Density = step(t, Age, Density, Kernel, Kmax, delta_t, delta_x,s1,s2,xmin,xmax,J)
        if compt == ecart:
            compt = 0
            Liste += [Density]
            K += [[Kernel(ts,x,Kmax,s1,s2,xmin,xmax) for x in Age]]
            x_obs,index_obs = calcul_x_obs(Kernel,ts,Age,Kmax,s1,s2,xmin,xmax,eps)
            mean = calcul_mean(Density[0:index_obs+1],Age[0:index_obs+1])
            Mean += [mean]
            Sd += [np.sqrt(calcul_variance(Density[0:index_obs+1], Age[0:index_obs+1], mean))]
            Obs += [x_obs]
            Current_time += [ts]
        else :
            compt += 1
        
    
    
    fig, ax = plt.subplots(figsize=(12,10))
    plt.xlim(0,20)
    plt.ylim(0,2.5)
    plt.grid()
    plt.xlabel("Age")
    plt.ylabel("Value")
    plt.suptitle("Evolution of the density and the kernel over time t, \n T = %s , X = %s , delta_t = %s , delta_x = %s , I = %s , J = %s  ,\n Kmax = %s, s1 = %s , s2 = %s , xmin = %s , xmax = %s, mu = %s , sigma = %s"
              %(T,X,round(delta_t,5),round(delta_x,5),I,J,Kmax,s1,s2,xmin,xmax,mu,sigma), fontsize=18)
    plt.legend()


    line, = ax.plot([],[], 'r--' , label = "Kernel")
    point, = ax.plot([],[], 'b', label = "Density")
    verti_mean, = ax.plot([],[], 'g--' , label = "Mean")
    verti_sd_left, = ax.plot([],[], color = "dimgrey", linestyle = '--' , label = "+1sigma")
    verti_sd_right, = ax.plot([],[], color = "dimgrey", linestyle = '--' , label = "-1sigma")
    verti_obs, = ax.plot([],[], color = "black", linestyle = '-' , label = "x_obs")

    def init():
        line.set_data([], [])
        point.set_data([], [])
        verti_mean.set_data([], [])
        verti_sd_left.set_data([], [])
        verti_sd_right.set_data([], [])
        verti_obs.set_data([], [])
        plt.legend()
        return (line,point,verti_mean,verti_sd_left,verti_sd_right,verti_obs)

    def animate(i):
        t = Time[i]
        x = Age
        y = Liste[i]
        z = K[i]
        m = Mean[i]
        sig = Sd[i]
        x_obs = Obs[i]
        line.set_data(x,z)
        point.set_data(x,y)
        verti_mean.set_data([m,m],[0,Kmax])
        verti_sd_left.set_data([m-sig,m-sig],[0,Kmax])
        verti_sd_right.set_data([m+sig,m+sig],[0,Kmax])
        verti_obs.set_data([x_obs,x_obs],[0,Kmax])
        plt.title("Current time =%s"%(round(t,2)),fontsize = 16)
        plt.legend()
        return (line,point,verti_mean,verti_sd_left,verti_sd_right,verti_obs)

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                             frames=len(Liste), interval=100, blit=True,
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



def graphwindow_mean_sd(T, X, delta_x, initial_n, Kernel, Kmax,s1,s2,xmin,xmax,mu,sigma,eps_kernel,eps_density):
    """
    Graph of the mean and standard deviation over time when almost all cells are in the window of observation

    Parameters
    ----------
    T : float
        Maximum time.
    X : float
        Maximum age.
    delta_x : float
        Age step.
    initial_n : function
        Description of our initial density. Takes two float inputs (x,X_older) and return a float.
    Kernel : function
        Kernel of transition taking four float inputs (t,x,X_older,Kmax) and return a float.
    Kmax : float
        Upper bound of the kernel.
    s1,s2,xmin,xmax,mu,sigma : float
        Parameters for our initial condition and for the Kernel.


    Returns
    -------
    Graph of the approximated mean and standard deviation over time when almost all cells are in the window of observation


    """
    plt.close()
    delta_t,delta_x,I,J,Time,Age = delta(T,X,delta_x,Kmax)
    x_obs,index_obs = calcul_x_obs(Kernel,Time[0],Age,Kmax,s1,s2,xmin,xmax,eps_kernel)
    x_end_obs,index_end_obs = calcul_x_end_obs(Kernel,Time[0],Age,Kmax,s1,s2,xmin,xmax,eps_kernel)
    
    Density = np.array([initial_n(x,mu,sigma) for x in Age])
    mean = calcul_mean(Density[0:index_obs+1],Age[0:index_obs+1])
    Time_obs = [Time[0]]
    Mean = [mean]
    Sd = [np.sqrt(calcul_variance(Density[0:index_obs+1], Age[0:index_obs+1], mean))]
        
        
    for i in range(len(Time)-1): # if we consider t = Time[-1], 
                         # we will calculate an approximation for a time bigger than T 
        t = Time[i]
        ts = Time[i+1]
        Density = step(t, Age, Density, Kernel, Kmax, delta_t, delta_x,s1,s2,xmin,xmax,J)
        x_obs,index_obs = calcul_x_obs(Kernel,ts,Age,Kmax,s1,s2,xmin,xmax,eps_kernel)
        x_end_obs,index_end_obs = calcul_x_end_obs(Kernel,ts,Age,Kmax,s1,s2,xmin,xmax,eps_kernel)
        if max(Density[index_obs:index_end_obs+1]) < eps_density and Density[0] < eps_density:
            mean = calcul_mean(Density[0:index_obs+1],Age[0:index_obs+1])
            Time_obs += [ts]
            Mean += [mean]
            Sd += [np.sqrt(calcul_variance(Density[0:index_obs+1], Age[0:index_obs+1],mean))]
        
    plt.scatter(Time_obs,Mean,marker = '.',s=2,label="Mean")
    plt.scatter(Time_obs,Sd,marker='.',s=2,label="Standard deviation")
    plt.xlim(0,T)
    plt.suptitle("Approximated mean and standard deviation over time ",fontsize = 30)
    plt.title("T=%s , X=%s , delta_x=%s , delta_t=%s , I= %s , J=%s ,\n Kmax=%s , xmin=%s , xmax=%s , s1=%s , s2=%s ,\n mu=%s , sigma=%s"%(T,X, round(delta_x,5), round(delta_t,5),I,J, Kmax, xmin,xmax,s1,s2,mu,sigma),fontsize = 20)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.show()
    
def regression_sd(T, X, delta_x, initial_n, Kernel, Kmax,s1,s2,xmin,xmax,mu,sigma,eps_kernel,eps_density):
    """
    Graph of the mean over time

    Parameters
    ----------
    T : float
        Maximum time.
    X : float
        Maximum age.
    delta_x : float
        Age step.
    initial_n : function
        Description of our initial density. Takes two float inputs (x,X_older) and return a float.
    Kernel : function
        Kernel of transition taking four float inputs (t,x,X_older,Kmax) and return a float.
    Kmax : float
        Upper bound of the kernel.
    s1,s2,xmin,xmax,mu,sigma : float
        Parameters for our initial condition and for the Kernel.


    Returns
    -------
    Graph of the approximated mean over time

    """
    plt.close()
    delta_t,delta_x,I,J,Time,Age = delta(T,X,delta_x,Kmax)
    x_obs,index_obs = calcul_x_obs(Kernel,Time[0],Age,Kmax,s1,s2,xmin,xmax,eps_kernel)
    x_end_obs,index_end_obs = calcul_x_end_obs(Kernel,Time[0],Age,Kmax,s1,s2,xmin,xmax,eps_kernel)
    
    Density = np.array([initial_n(x,mu,sigma) for x in Age])
    mean = calcul_mean(Density[0:index_obs+1],Age[0:index_obs+1])
    Time_obs = [Time[0]]
    Sd = [np.sqrt(calcul_variance(Density[0:index_obs+1], Age[0:index_obs+1], mean))]
        
        
    for i in range(len(Time)-1): # if we consider t = Time[-1], 
                         # we will calculate an approximation for a time bigger than T 
        t = Time[i]
        ts = Time[i+1]
        Density = step(t, Age, Density, Kernel, Kmax, delta_t, delta_x,s1,s2,xmin,xmax,J)
        x_obs,index_obs = calcul_x_obs(Kernel,ts,Age,Kmax,s1,s2,xmin,xmax,eps_kernel)
        x_end_obs,index_end_obs = calcul_x_end_obs(Kernel,ts,Age,Kmax,s1,s2,xmin,xmax,eps_kernel)
        if max(Density[index_obs:index_end_obs+1]) < eps_density and Density[0] < eps_density:
            mean = calcul_mean(Density[0:index_obs+1],Age[0:index_obs+1])
            Time_obs += [ts]
            Sd += [np.sqrt(calcul_variance(Density[0:index_obs+1], Age[0:index_obs+1],mean))]
    
    
    t_jump = Time[0]
    Time_jump = [t_jump] # Between Time_jump[2*i] and Time_jump[2*i+1] we have calculated sd
    # no sd calculated between Time_jump[2*i+1] and Time_jump[2*i+2] (too much cells can still transitionned)
    Index_jump = [0]
    L = np.size(Time_obs)
    for l in range(L-1):
        if Time_obs[l+1]-Time_obs[l] > 1.5*delta_t : # if we miss a time, there is a gap
            Time_jump += [Time_obs[l],Time_obs[l+1]]
            Index_jump += [l,l+1]
    Time_jump += [Time_obs[-1]]
    Index_jump += [L-1]
        
        
    
    plt.scatter(Time_obs,Sd,color = 'r',marker='.',s=1,label="Standard deviation")
    plt.vlines(Time_jump,ymin=0,ymax=X,linestyles = 'dashed',label="Limits of each section")
    plt.plot([],[],color = 'b' , label = 'Linear regression')
    for i in range(int(np.size(Index_jump)/2)):
        index_begin_section = Index_jump[2*i]
        index_end_section = Index_jump[2*i+1]
        x = Time_obs[index_begin_section:index_end_section+1]
        y = Sd[index_begin_section:index_end_section+1]
        a , b = np.polyfit(x, y, 1) # linear regression of the form a*X + b = Y
        t0 = Time_obs[index_begin_section]
        t1 = Time_obs[index_end_section]
        y0 = a*t0+b
        y1 = a*t1+b
        plt.plot([t0,t1],[y0,y1],color = 'b')
        
    plt.xlim(-0.1,T+0.1)
    plt.ylim(0,1.1*max(Sd))
    plt.suptitle("Approximated standard deviation over time, with linear regression ", fontsize = 24)
    plt.title("T=%s , X=%s , delta_x=%s , delta_t=%s , I= %s , J=%s ,\n Kmax=%s , xmin=%s , xmax=%s , s1=%s , s2=%s ,\n mu=%s , sigma=%s"%(T,X, round(delta_x,5), round(delta_t,5),I,J, Kmax, xmin,xmax,s1,s2,mu,sigma),fontsize = 20)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.show()
    
def slope_sd(T, X, delta_x, initial_n, Kernel, Kmax,s1,s2,xmin,xmax,mu,sigma,eps_kernel,eps_density):
    """
    Graph of the slope and y-intercept of the approximated standard deviation of each section observe over time

    Parameters
    ----------
    T : float
        Maximum time.
    X : float
        Maximum age.
    delta_x : float
        Age step.
    initial_n : function
        Description of our initial density. Takes two float inputs (x,X_older) and return a float.
    Kernel : function
        Kernel of transition taking four float inputs (t,x,X_older,Kmax) and return a float.
    Kmax : float
        Upper bound of the kernel.
    s1,s2,xmin,xmax,mu,sigma : float
        Parameters for our initial condition and for the Kernel.


    Returns
    -------
    Graph of the slope and y-intercep of the approximated standard deviation of each section observe over time

    """
    plt.close()
    delta_t,delta_x,I,J,Time,Age = delta(T,X,delta_x,Kmax)
    x_obs,index_obs = calcul_x_obs(Kernel,Time[0],Age,Kmax,s1,s2,xmin,xmax,eps_kernel)
    x_end_obs,index_end_obs = calcul_x_end_obs(Kernel,Time[0],Age,Kmax,s1,s2,xmin,xmax,eps_kernel)
    
    Density = np.array([initial_n(x,mu,sigma) for x in Age])
    mean = calcul_mean(Density[0:index_obs+1],Age[0:index_obs+1])
    Time_obs = [Time[0]]
    Sd = [np.sqrt(calcul_variance(Density[0:index_obs+1], Age[0:index_obs+1], mean))]
        
        
    for i in range(len(Time)-1): # if we consider t = Time[-1], 
                         # we will calculate an approximation for a time bigger than T 
        t = Time[i]
        ts = Time[i+1]
        Density = step(t, Age, Density, Kernel, Kmax, delta_t, delta_x,s1,s2,xmin,xmax,J)
        x_obs,index_obs = calcul_x_obs(Kernel,ts,Age,Kmax,s1,s2,xmin,xmax,eps_kernel)
        x_end_obs,index_end_obs = calcul_x_end_obs(Kernel,ts,Age,Kmax,s1,s2,xmin,xmax,eps_kernel)
        if max(Density[index_obs:index_end_obs+1]) < eps_density and Density[0] < eps_density:
            mean = calcul_mean(Density[0:index_obs+1],Age[0:index_obs+1])
            Time_obs += [ts]
            Sd += [np.sqrt(calcul_variance(Density[0:index_obs+1], Age[0:index_obs+1],mean))]
    
    
    t_jump = Time[0]
    Time_jump = [t_jump] # Between Time_jump[2*i] and Time_jump[2*i+1] we have calculated sd
    # no sd calculated between Time_jump[2*i+1] and Time_jump[2*i+2] (too much cells can still transitionned)
    Index_jump = [0]
    L = np.size(Time_obs)
    for l in range(L-1):
        if Time_obs[l+1]-Time_obs[l] > 1.5*delta_t : # if we miss a time, there is a gap
            Time_jump += [Time_obs[l],Time_obs[l+1]]
            Index_jump += [l,l+1]
    Time_jump += [Time_obs[-1]]
    Index_jump += [L-1]
    
    

        
        
    Section = []
    Slope = []
    Intercept = []
    
    for i in range(int(np.size(Index_jump)/2)):
        index_begin_section = Index_jump[2*i]
        index_end_section = Index_jump[2*i+1]
        x = Time_obs[index_begin_section:index_end_section+1]
        y = Sd[index_begin_section:index_end_section+1]
        a , b = np.polyfit(x, y, 1) # linear regression of the form a*X + b = Y
        Section += [i]
        Slope += [a]
        Intercept += [b]
    
    fig,ax = plt.subplots(1,2)
    
    plt.ylim(0,1.1*max(max(Slope),max(Intercept)))
    ax[0].scatter(Section,Slope,label="Slope")
    ax[0].legend()
    ax[1].scatter(Section,Intercept,label="Y-intercept")
    ax[1].legend()
    plt.suptitle("Slopes and y-intercept of observed section \n \n T=%s , X=%s , delta_x=%s , delta_t=%s , I= %s , J=%s ,\n Kmax=%s , xmin=%s , xmax=%s , s1=%s , s2=%s ,\n mu=%s , sigma=%s"%(T,X, round(delta_x,5), round(delta_t,5),I,J, Kmax, xmin,xmax,s1,s2,mu,sigma),fontsize = 20)
    ax[0].set_xlabel("Section")
    ax[0].set_ylabel("Value")
    ax[1].set_xlabel("Section")
    ax[1].set_ylabel("Value")
    plt.show()
    
