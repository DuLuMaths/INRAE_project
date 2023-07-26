#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 7 2023

@author: ldumetz

Creation of files for the numerical exploration for the 2 phases circadian clock system
"""

import numpy as np
import scipy as sp
from joblib import Parallel, delayed
from scipy.stats import qmc



# Creation of the initial condition as a gaussian

def initialn_p(x,p,Parameters_initial):
    c,mu,sigma = Parameters_initial[p]
    result = c*(np.exp(-(x-mu)**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma))
    return(result)

def vect_density_init(Parameters_initial,List_Age):
    Density_init = []
    Density_init += [initialn_p(x,0,Parameters_initial) for x in List_Age]
    Density_init += [initialn_p(x,1,Parameters_initial) for x in List_Age]
    Density = np.array(Density_init)
    return(Density)

# Creation of the kernel of transition as an approximated simple function

def kernel_p(t,x,p,Mass_phases,delta_x,List_Kmax,Parameters_fixed_kernel,Parameters_play_kernel): 
    Kmaxp = List_Kmax[p]
    sl,sr,xmin,xmax = Parameters_fixed_kernel[p]
    a , b = Parameters_play_kernel[p]
    if p == 0:
        mass_self , mass_other = Mass_phases[0] , Mass_phases[1]
    else :
        mass_self , mass_other = Mass_phases[1] , Mass_phases[0]
    coeff = a*mass_other+b*mass_self+Kmaxp
    result = coeff*(np.tanh(sl*(x-xmin))+np.tanh(sr*(xmax-x)))/2
    return(result)

# Since we work with P=2 phases, we want to be able to take them apart
# Indeed, we will work with Density a vector that is the concatenation of 
# the approximated density in each phase

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
    return(Subvectors)

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

def delta(T,X,delta_x,Parameters_play_kernel,error_mass,List_Kmax):
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
    a1,b1 = Parameters_play_kernel[0]
    a2,b2 = Parameters_play_kernel[1]
    Kmax1,Kmax2 = List_Kmax
    List_Age = np.linspace(0, X,1000)
    Density_init = vect_density_init(Parameters_initial,List_Age)
    Mass_phase , mass_tot = mass(Density_init,2,delta_x)
    mass_max = mass_tot*(1+error_mass)
    maxK1 = max(a1*mass_max+Kmax1,b1*mass_max+Kmax1)
    maxK2 = max(a2*mass_max+Kmax1,b2*mass_max+Kmax1)
    Kmax = max(maxK1,maxK2)
    delta_t =(delta_x/(1+Kmax*delta_x))/1.01
    I = int(T/delta_t)
    J = int(X/delta_x)
    # the choice of I and J impose that the CFL conditions are still true
    Time = np.array([i*delta_t for i in range(I+1)]) # discretisation of our time
    Age = np.array([j*delta_x for j in range(J+1)]) # discretisation of our age
    
    return(delta_t,delta_x,I,J,Time,Age)
        
    
def matrix_transition(t,Age,Mass_phases,delta_t,delta_x,kernel_p,List_Kmax,Parameters_fixed_kernel,Parameters_play_kernel):
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
    Data += [(delta_x/2)*kernel_p(t,Age[0],P-1,Mass_phases,delta_x,List_Kmax,Parameters_fixed_kernel,Parameters_play_kernel)]
    Data += [delta_x*kernel_p(t,Age[j],P-1,Mass_phases,delta_x,List_Kmax,Parameters_fixed_kernel,Parameters_play_kernel) for j in range(1,L-1)]
    Data += [(delta_x/2)*kernel_p(t,Age[-1],P-1,Mass_phases,delta_x,List_Kmax,Parameters_fixed_kernel,Parameters_play_kernel)]
    
    Row += [l for l in range(1,L)]
    Col += [l for l in range(L-1)]
    Data += [ratio for l in range(L-1)]
    
    Row += [l for l in range(1,L)]
    Col += [l for l in range(1,L)]
    Data += [1-ratio-delta_t*kernel_p(t,Age[l],0,Mass_phases,delta_x,List_Kmax,Parameters_fixed_kernel,Parameters_play_kernel) for l in range(1,L)]
    
    
    for p in range(1,P):
        Row += [p*L for l in range(L)]
        Col += [(p-1)*L+l for l in range(L)]
        Data += [(delta_x/2)*kernel_p(t,Age[0],p-1,Mass_phases,delta_x,List_Kmax,Parameters_fixed_kernel,Parameters_play_kernel)]
        Data += [delta_x*kernel_p(t,Age[j],p-1,Mass_phases,delta_x,List_Kmax,Parameters_fixed_kernel,Parameters_play_kernel) for j in range(1,L-1)]
        Data += [(delta_x/2)*kernel_p(t,Age[-1],p-1,Mass_phases,delta_x,List_Kmax,Parameters_fixed_kernel,Parameters_play_kernel)]
        
        Row += [l+p*L for l in range(1,L)]
        Col += [l+p*L for l in range(L-1)]
        Data += [ratio for l in range(L-1)]
        
        Row += [l+p*L for l in range(1,L)]
        Col += [l+p*L for l in range(1,L)]
        Data += [1-ratio-delta_t*kernel_p(t,Age[l],p,Mass_phases,delta_x,List_Kmax,Parameters_fixed_kernel,Parameters_play_kernel) for l in range(1,L)]
        
    row = np.array(Row)
    col = np.array(Col)
    data = np.array(Data)
    mtx = sp.sparse.coo_matrix((data, (row, col)), shape=(L*P, L*P))
    
    return(mtx)



def step(t,Density_old,Age,delta_t,delta_x,kernel_p,List_Kmax,Parameters_fixed_kernel,Parameters_play_kernel):
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
    Mass_phases , Mass_tot = mass(Density_old,2,delta_x)
    M = matrix_transition(t,Age,Mass_phases,delta_t,delta_x,kernel_p,List_Kmax,Parameters_fixed_kernel,Parameters_play_kernel)
    Density = M@Density_old
    return(Density,Mass_phases,Mass_tot)

def choice_points(List_Kmax,X,Parameters_initial,error_mass,maxbound):
    List_Age = np.linspace(0, X,1000)
    Density_init = vect_density_init(Parameters_initial,List_Age)
    Mass_phase , mass_tot = mass(Density_init,2,delta_x)
    mass_max = mass_tot*(1+error_mass)
    alpha1 = List_Kmax[0]/mass_max
    alpha2 = List_Kmax[1]/mass_max
    
    

    sampler = qmc.Sobol(d=4, scramble=False) # d = dimension of the cube
    Sample_unit = sampler.random_base2(m=2) # 2^m = number of points in the [0,1) d-cube

    # we want to explore each zone of the 4-cube
    # precisely : we want to have points where only one parameter is negative,
    # points where two parameters are negative, and so on (for all parameters)

    # 1 means the parameter is negative, 0 it is non-negative
    
    #initialisation
    u_bounds = [maxbound,maxbound,maxbound,maxbound]
    l_bounds = [0,0,0,0]
    Sample = qmc.scale(Sample_unit, l_bounds, u_bounds)

    
    for a in [0,1]:
        for b in [0,1]:
            for c in [0,1]:
                for d in [0,1]:
                    if not(a or b or c or d):
                        #do nothing, it the initialisation
                        continue
                    else :
                        if a:
                            low_a = -alpha1
                            upper_a = 0
                        else :
                            low_a = 0
                            upper_a = maxbound
                        if b:
                            low_b = -alpha1
                            upper_b = 0
                        else :
                            low_b = 0
                            upper_b = maxbound
                        if c:
                            low_c = -alpha2
                            upper_c = 0
                        else :
                            low_c = 0
                            upper_c = maxbound
                        if d:
                            low_d = -alpha2
                            upper_d = 0
                        else :
                            low_d = 0
                            upper_d = maxbound
                        u_bounds = [upper_a,upper_b,upper_c,upper_d]
                        l_bounds = [low_a,low_b,low_c,low_d]
                        s = qmc.scale(Sample_unit, l_bounds, u_bounds)
                        Sample = np.concatenate((Sample,s),axis=0)
  
    
    Separate = []
    for s in Sample:
        a1 , b1 , a2 , b2 = s
        memo = [[a1,b1],[a2,b2]]
        Separate += [memo]
    Resultat = np.array(Separate)
    return(Sample,Resultat)

# def choice_points(List_Kmax,X,Parameters_initial,error_mass,size_4cube):
#     List_Age = np.linspace(0, X,1000)
#     Density_init = vect_density_init(Parameters_initial,List_Age)
#     Mass_phase , mass_tot = mass(Density_init,2,delta_x)
#     mass_max = mass_tot*(1+error_mass)
#     alpha1 = List_Kmax[0]/mass_max
#     alpha2 = List_Kmax[1]/mass_max
    
#     sampler = qmc.Sobol(d=4, scramble=False) # d = dimension of the cube
#     Sample = sampler.random_base2(m=2) # 2^m = number of points in the [0,1) d-cube
#     Sample = size_4cube*Sample
#     Sample = Sample - [alpha1,alpha1,alpha2,alpha2]
#     Sample = np.concatenate((Sample,np.array([[0,0,0,0]])),axis=0)
#     Separate = []
#     for s in Sample:
#         a1 , b1 , a2 , b2 = s
#         memo = [[a1,b1],[a2,b2]]
#         Separate += [memo]
#     Resultat = np.array(Separate)
#     return(Sample,Resultat)



def compute_pde(par,init_data):
    M1 = []
    M2 = []
    N1 = []
    N2 = []
    T_density = []
    T,X,time_memo,List_Kmax,delta_x,Parameters_fixed_kernel,Parameters_initial,size_4cube = init_data
    Parameters_play_kernel = par
    delta_t,delta_x,I,J,Time,Age = delta(T,X,delta_x,Parameters_play_kernel,error_mass,List_Kmax)
    # Creation of the vector which is the concatenation of the initial density in each phase
    Density = vect_density_init(Parameters_initial,Age)
    
    ecart = 0
    while Time[ecart] <= time_memo:
        ecart += 1
    
    compt = 0
    Mass_phases,mass_tot = mass(Density, 2, delta_x)
    M1 += [Mass_phases[0]]
    M2 += [Mass_phases[1]]
    Subvectors = separation(Density,2)
    N1 += [Subvectors[0]]
    N2 += [Subvectors[1]]
    T_density += [Time[0]]
    for i in range(len(Time)-1): # if we consider t = Time[-1], 
                         # we will calculate an approximation for a time bigger than T 
        t = Time[i]
        ts = Time[i+1]
        Density,Mass_phases,mass_tot = step(t,Density,Age,delta_t,delta_x,kernel_p,List_Kmax,Parameters_fixed_kernel,Parameters_play_kernel)
        M1 += [Mass_phases[0]]
        M2 += [Mass_phases[1]]
        if compt == ecart:
            compt = 0
            Subvectors = separation(Density,2)
            N1 += [Subvectors[0]]
            N2 += [Subvectors[1]]
            T_density += [ts]
        else :
            compt += 1
    return(M1,M2,Time,Age,N1,N2,T_density)


if __name__== "__main__":
    
        
    #-----------------------------------------------------
    # PARAMETRES EN DUR (a determiner en fonction de la definition des phases)
    # A rentrer en dur dans les fonctions
    
    # Parameters of our scheme
    T = 200 # final time
    X = 18 # maximum age such that we don't miss information in true differential system
    n_proc = 5 # Number of processor use for the parallelisation
    time_memo = 1/6 # To gain space, we will memorise each density when about time_memo has passed
                    # If 1 unit of time represents 1 hour, 
                    # then time_memo = 1/12 means we look at the density every 5 min;
                    # 1/6 means every 10 min
    maxbound = 2 # for the sampling, we want to sample a 4-cube of a given size 
    
    List_Kmax = [0.3 , 0.3] # List of maximum values of the kernel
    delta_x = 1/40 # Age step wanted
    error_mass = 1/200 # Margin of error for the total mass 
    Parameters_fixed_kernel = [[8,8,8,15], [8,8,5,15]]
    # If each kernel is an approximation a a simple function independant of time, 
    # we need a multiplicative constance, a slope to the left, a slope to the right, 
    # a minimum age where transition starts and a maximum age where the transition ends
    
    
    # Initialization IBV
    Parameters_initial = [[1.5,5,1],[1,2.5,0.5]]
    # If each initial conditions are Gaussian, we need to specify the scaling factor
    # the mean and the standard deviation
    
    #------------------------------------------------------
    # Inputs in order to compute in parallel
    # Parameters_play_kernel = [[0.8,-0.1],[0.8,-0.1]] # For the affine coupling
    # In order to choose wisely the set of parameters tuples use sobol sequence
    

    
    #------------------------------------------------------
    # Launch parallel code
    # toto = sampling of set of parameters by sobol sequence
    # n_it = len(toto[:,0])
    # init_data = initial conditions 
    # n_proc = 70
    
    goodarray , toto = choice_points(List_Kmax,X,Parameters_initial,error_mass,maxbound)
    n_it = len(toto[:,0])
    
    init_data = [T,X,time_memo,List_Kmax,delta_x,Parameters_fixed_kernel,Parameters_initial,maxbound]
    
    def pde_loop(par, ind_proc):
        M1,M2,Time,Age,N1,N2,T_density = compute_pde(par,init_data)
        
        np.savetxt('./repositery/parameters_play_it_'+ str(ind_proc)+'.txt',par)
        np.savetxt('./repositery/mass_1_it_'+ str(ind_proc)+'.txt',M1)
        np.savetxt('./repositery/mass_2_it_'+ str(ind_proc)+'.txt',M2)
        np.savetxt('./repositery/time_mass_it_'+ str(ind_proc)+'.txt',Time)
        np.savetxt('./repositery/age_mass_it_'+ str(ind_proc)+'.txt',Age)
        np.savetxt('./repositery/density_1_it_'+ str(ind_proc)+'.txt',N1)
        np.savetxt('./repositery/density_2_it_'+ str(ind_proc)+'.txt',N2)
        np.savetxt('./repositery/time_density_it_'+ str(ind_proc)+'.txt',T_density)
        print('Done for sim number '+ str(ind_proc))
        return()
    
    def array_init(init_data):
        T,X,time_memo,List_Kmax,delta_x,Parameters_fixed_kernel,Parameters_initial,size_4cube = init_data
        np.savetxt('./repositery/T_X_timememo_deltax_sizecube'+'.txt',[T,X,time_memo,delta_x,size_4cube])
        np.savetxt('./repositery/List_kmax_'+'.txt',List_Kmax)
        np.savetxt('./repositery/Parameters_fixed_kernel_it_'+'.txt',Parameters_fixed_kernel)
        np.savetxt('./repositery/Parameters_initial_it_'+'.txt',Parameters_initial)
        return()
    

    np.savetxt('./repositery/Sobol'+'.txt',goodarray)
    array_init(init_data)
    Parallel(n_jobs=n_proc)(delayed(pde_loop)(toto[ie,:],ie) for ie in range(n_it))
