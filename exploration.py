#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 10:15:05 2023

@author: ldumetz

Numerical exploration for the 2 phases circadian clock system
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from joblib import Parallel, delayed
from numpy.fft import fft, fftfreq , irfft
import matplotlib.colors as mcolors
import matplotlib.cm as cm

def extract_sobol():
    fichier_sobol = open("Sobol.txt", "r")
    Sobol_phase = []
    for l in fichier_sobol.readlines():
        L = []
        for s in l.split():
            L += [float(s)]
        Sobol_phase += [L]
    return(Sobol_phase)

def extract_para():
    fichier_data = open("T_X_timememo_deltax_sizecube.txt", "r")
    Forstep = []
    
    for l in fichier_data.readlines():
        Forstep += [float(l)]
        
    T, X, time_memo, delta_x, maxbound = Forstep
    
    #______________________________________________________________
    
    fichier_kmax = open("List_kmax_.txt", "r")
    List_Kmax = []
    for l in fichier_kmax.readlines():
        List_Kmax += [float(l)]
        
    #______________________________________________________________   
        
    fichier_para_fixed = open("Parameters_fixed_kernel_it_.txt", "r")
    l1_para_fixed , l2_para_fixed = fichier_para_fixed.readlines()
    Parameters_fixed_kernel = []
    L1_fixed , L2_fixed = [] , []
    for s in l1_para_fixed.split() :
        L1_fixed += [float(s)]
    for s in l2_para_fixed.split() :
        L2_fixed += [float(s)]
    Parameters_fixed_kernel += [L1_fixed,L2_fixed]
    
    #______________________________________________________________   
        
    fichier_para_init = open("Parameters_initial_it_.txt", "r")
    l1_para_init , l2_para_init = fichier_para_init.readlines()
    Parameters_init = []
    L1_init , L2_init = [] , []
    for s in l1_para_init.split() :
        L1_init += [float(s)]
    for s in l2_para_init.split() :
        L2_init += [float(s)]
    Parameters_init += [L1_init,L2_init]
    
    
    return(T, X, time_memo, delta_x, maxbound, List_Kmax, Parameters_fixed_kernel, Parameters_init)


def extract_info(number):
    fichier_data = open("parameters_play_it_%s.txt"%(number), "r")
    l1 , l2 = fichier_data.readlines()
    Para1 , Para2 = [] , []
    for s in l1.split():
        Para1 += [float(s)]
    for s in l2.split():
        Para2 += [float(s)]
    Parameters_play_kernel = [Para1,Para2]
    
    #______________________________________________________________  
    
    fichier_age = open("age_mass_it_%s.txt"%(number), "r")
    Age = []
    for l in fichier_age.readlines():
        Age += [float(l)]
    
    #______________________________________________________________  
    
    fichier_density1 = open("density_1_it_%s.txt"%(number), "r")
    Density1 = []
    for l in fichier_density1.readlines():
        L = []
        for s in l.split():
            L += [float(s)]
        Density1 += [L]
        
    #______________________________________________________________  
     
    fichier_density2 = open("density_2_it_%s.txt"%(number), "r")
    Density2 = []
    for l in fichier_density2.readlines():
        L = []
        for s in l.split():
             L += [float(s)]
        Density2 += [L]
        
    #______________________________________________________________  
    
    fichier_mass1 = open("mass_1_it_%s.txt"%(number), "r")
    Mass1 = []
    for l in fichier_mass1.readlines():
        Mass1 += [float(l)]
    
    #______________________________________________________________  
    
    fichier_mass2 = open("mass_2_it_%s.txt"%(number), "r")
    Mass2 = []
    for l in fichier_mass2.readlines():
        Mass2 += [float(l)]
        
    #______________________________________________________________  
    
    fichier_time_density = open("time_density_it_%s.txt"%(number), "r")
    Time_density = []
    for l in fichier_time_density.readlines():
        Time_density += [float(l)]
        
    #______________________________________________________________  
    
    fichier_time = open("time_mass_it_%s.txt"%(number), "r")
    Time = []
    for l in fichier_time.readlines():
        Time += [float(l)]
    
    return(Parameters_play_kernel,Age,Time,Mass1,Mass2,Time_density,Density1,Density2)


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


def animation_density_separated(number):
    T, X, time_memo, delta_x, maxbound, List_Kmax, Parameters_fixed_kernel, Parameters_init = extract_para()
    Parameters_play_kernel,Age,Time,Mass1,Mass2,Time_density,Density1,Density2 = extract_info(number)
    delta_t = Time[1]-Time[0]
    I = np.size(Time)
    J = np.size(Age)
    Kmax1 , Kmax2 = List_Kmax
    sl1,sr1,xmin1,xmax1 = Parameters_fixed_kernel[0]
    sl2,sr2,xmin2,xmax2 = Parameters_fixed_kernel[1]
    c1,mu1,sigma1 = Parameters_init[0]
    c2,mu2,sigma2 = Parameters_init[1]
    a1,b1 = Parameters_play_kernel[0]
    a2,b2 = Parameters_play_kernel[1]
    M_tot = Mass1[0]+Mass2[0]
    val_min1 = -Kmax1/M_tot
    val_min2 = -Kmax2/M_tot
    
    
    ecart = 0
    while Time[ecart] <= time_memo:
        ecart += 1
    
    Memory_K1 = [[kernel_p(Time[0],x,0,Mass1[0],Mass2[0],List_Kmax,Parameters_fixed_kernel,Parameters_play_kernel) for x in Age]]
    Memory_K2 = [[kernel_p(Time[0],x,1,Mass1[0],Mass2[0],List_Kmax,Parameters_fixed_kernel,Parameters_play_kernel) for x in Age]]
    
    compt = 0
    
    for i in range(len(Time)-1):
        ts = Time[i+1]
        if compt == ecart:
            compt = 0
            Memory_K1 += [[kernel_p(ts,x,0,Mass1[i+1],Mass2[i+1],List_Kmax,Parameters_fixed_kernel,Parameters_play_kernel) for x in Age]]
            Memory_K2 += [[kernel_p(ts,x,1,Mass1[i+1],Mass2[i+1],List_Kmax,Parameters_fixed_kernel,Parameters_play_kernel) for x in Age]]
        else :
            compt += 1
    
    
    fig, (ax0,ax1) = plt.subplots(2,1,figsize=(12,10))
    ax0.set_ylim(0, 1.2)
    ax0.set_xlim(0, 1.01*X)
    ax0.set_xlabel("Age")
    ax0.set_ylabel("Value")
    ax0.grid()

    
    ax1.set_ylim(0, 1.2)
    ax1.set_xlim(0, 1.01*X)
    ax1.set_xlabel("Age")
    ax1.set_ylabel("Value")
    ax1.grid()

            
    plt.suptitle("Evolution of the density and the kernel over time t \n T = %s , X = %s , delta_t = %s , delta_x = %s , I = %s , J = %s , \n Kmax1 = %s , sl1 = %s , sr1 = %s , xmin1 = %s , xmax1 = %s \n Kmax2 = %s , sl2 = %s , sr2 = %s , xmin2 = %s , xmax2 = %s , \n c1 = %s , mu1 = %s , sigma1 = %s, c2 = %s , mu2 = %s , sigma2 = %s \n (Minimum value for a1 and b1 = %s) a1 = %s , b1 = %s , (Minimum value for a2 and b2 = %s) a2 = %s , b2 = %s"
              %(T,X,round(delta_t,5),round(delta_x,5),I,J,round(Kmax1,4),round(sl1,4),round(sr1,4),round(xmin1,4),round(xmax1,4),round(Kmax2,4),round(sl2,4),round(sr2,4),round(xmin2,4),round(xmax2,4),round(c1,4),round(mu1,4),round(sigma1,4),round(c2,4),round(mu2,4),round(sigma2,4),round(val_min1,4),round(a1,4),round(b1,4),round(val_min2,4),round(a2,4),round(b2,4)), fontsize=10)
    

    densityani1, = ax0.plot([],[], color = 'red' , label = "Density phase %s"%(1))
    densityani2, = ax1.plot([],[], color = 'blue' ,label = "Density phase %s"%(2))
    kernelani1, = ax0.plot([],[], color ='darkred' , linestyle = '--' , label = "Kernel from %s to %s"%(1,2))
    kernelani2, = ax1.plot([],[], color ='darkblue' , linestyle = '--' , label = "Kernel from %s to %s"%(1,2))


    def init():
        densityani1.set_data([], [])
        densityani2.set_data([], [])
        kernelani1.set_data([], [])
        kernelani2.set_data([], [])
        ax0.legend()
        ax1.legend()
        return (densityani1,densityani2,kernelani1,kernelani2)

    def animate(i):
        t = Time_density[i]
        plt.title("Current time =%s"%(round(t,2)),fontsize = 10)
        densityani1.set_data(Age,Density1[i])
        densityani2.set_data(Age,Density2[i])
        kernelani1.set_data(Age,Memory_K1[i])
        kernelani2.set_data(Age,Memory_K2[i])
        ax0.legend(loc = 'upper left')
        ax1.legend(loc = 'upper left')
        return (densityani1,densityani2,kernelani1,kernelani2)

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                             frames=len(Time_density), interval=100, blit=True,
                             repeat=True)
    
    anim.save(filename="Animation_density_%s.gif"%(number), dpi=70, fps=12, writer = 'pillow')
    
    
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
    
    #rc(animation)
    return()


def graph_mass(number):
    T, X, time_memo, delta_x, maxbound, List_Kmax, Parameters_fixed_kernel, Parameters_init = extract_para()
    Parameters_play_kernel,Age,Time,Mass1,Mass2,Time_density,Density1,Density2 = extract_info(number)
    delta_t = Time[1]-Time[0]
    I = np.size(Time)
    J = np.size(Age)
    Kmax1 , Kmax2 = List_Kmax
    sl1,sr1,xmin1,xmax1 = Parameters_fixed_kernel[0]
    sl2,sr2,xmin2,xmax2 = Parameters_fixed_kernel[1]
    c1,mu1,sigma1 = Parameters_init[0]
    c2,mu2,sigma2 = Parameters_init[1]
    a1,b1 = Parameters_play_kernel[0]
    a2,b2 = Parameters_play_kernel[1]
    M_tot = Mass1[0]+Mass2[0]
    val_min1 = -Kmax1/M_tot
    val_min2 = -Kmax2/M_tot
    
    Mass_tot = []
    for i in range(len(Mass1)):
        m = Mass1[i] + Mass2[i]
        Mass_tot += [m]
    
    fig, (ax0,ax1,axt) = plt.subplots(3,1,figsize=(12,10))
    ax0.set_xlabel("Time")
    ax0.set_ylabel("Mass")
    ax0.grid()
    
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Mass")
    ax1.grid()
    
    axt.set_xlabel("Time")
    axt.set_ylabel("Mass")
    axt.grid()
    
    plt.suptitle("Evolution of the mass over time t \n T = %s , X = %s , delta_t = %s , delta_x = %s , I = %s , J = %s , \n Kmax1 = %s , sl1 = %s , sr1 = %s , xmin1 = %s , xmax1 = %s \n Kmax2 = %s , sl2 = %s , sr2 = %s , xmin2 = %s , xmax2 = %s , \n c1 = %s , mu1 = %s , sigma1 = %s, c2 = %s , mu2 = %s , sigma2 = %s \n (Minimum value for a1 and b1 = %s) a1 = %s , b1 = %s , (Minimum value for a2 and b2 = %s) a2 = %s , b2 = %s"
              %(T,X,round(delta_t,5),round(delta_x,5),I,J,round(Kmax1,4),round(sl1,4),round(sr1,4),round(xmin1,4),round(xmax1,4),round(Kmax2,4),round(sl2,4),round(sr2,4),round(xmin2,4),round(xmax2,4),round(c1,4),round(mu1,4),round(sigma1,4),round(c2,4),round(mu2,4),round(sigma2,4),round(val_min1,4),round(a1,4),round(b1,4),round(val_min2,4),round(a2,4),round(b2,4)), fontsize=6)
    
    ax0.plot(Time,Mass1,label = "Phase 1")
    ax1.plot(Time,Mass2,label = "Phase 2")
    axt.plot(Time,Mass_tot,label = "Total")
    
    ax0.legend()
    ax1.legend()
    axt.legend()
    
    plt.savefig('Graph_mass_%s.png'%(number))
    return()

def graph_occupation_mean(number):
    """ The quantity (1/t) times the integral from 0 to t of the indicator function of M1(s) > M2(s) ds 
    represents, when t goes to infinity, the average time the population spend in state 1.
    The discrete version is (1/i+1) times the sum from k=0 to i of the indicator of M1(t_k) > M2(t_k)
    can be calculated.
    """
    plt.close()
    T, X, time_memo, delta_x, maxbound, List_Kmax, Parameters_fixed_kernel, Parameters_init = extract_para()
    Parameters_play_kernel,Age,Time,Mass1,Mass2,Time_density,Density1,Density2 = extract_info(number)
    delta_t = Time[1]-Time[0]
    I = np.size(Time)
    J = np.size(Age)
    Kmax1 , Kmax2 = List_Kmax
    sl1,sr1,xmin1,xmax1 = Parameters_fixed_kernel[0]
    sl2,sr2,xmin2,xmax2 = Parameters_fixed_kernel[1]
    c1,mu1,sigma1 = Parameters_init[0]
    c2,mu2,sigma2 = Parameters_init[1]
    a1,b1 = Parameters_play_kernel[0]
    a2,b2 = Parameters_play_kernel[1]
    M_tot = Mass1[0]+Mass2[0]
    val_min1 = -Kmax1/M_tot
    val_min2 = -Kmax2/M_tot
    
    
    S = 0
    X = []
    Y = []
    for i in range(len(Time)):
        if Mass1[i] >= Mass2[i]:
            S += 1
        X += [Time[i]]
        Y += [S/(i+1)]
    
    plt.plot(X,Y)
    
    plt.suptitle("Proportion of time spend in state 1 over time t \n T = %s , X = %s , delta_t = %s , delta_x = %s , I = %s , J = %s , \n Kmax1 = %s , sl1 = %s , sr1 = %s , xmin1 = %s , xmax1 = %s \n Kmax2 = %s , sl2 = %s , sr2 = %s , xmin2 = %s , xmax2 = %s , \n c1 = %s , mu1 = %s , sigma1 = %s, c2 = %s , mu2 = %s , sigma2 = %s \n (Minimum value for a1 and b1 = %s) a1 = %s , b1 = %s , (Minimum value for a2 and b2 = %s) a2 = %s , b2 = %s"
              %(T,X,round(delta_t,5),round(delta_x,5),I,J,round(Kmax1,4),round(sl1,4),round(sr1,4),round(xmin1,4),round(xmax1,4),round(Kmax2,4),round(sl2,4),round(sr2,4),round(xmin2,4),round(xmax2,4),round(c1,4),round(mu1,4),round(sigma1,4),round(c2,4),round(mu2,4),round(sigma2,4),round(val_min1,4),round(a1,4),round(b1,4),round(val_min2,4),round(a2,4),round(b2,4)), fontsize=6)
    
    
    
    plt.savefig('Graph_prop_%s.png'%(number))
    return()

    
def find3max(Ampli):
    i = 2
    x,y,z = Ampli[:3]
    if x >= y and x >= z:
        i_memo1 = 0
        max_ampli1 = x
        if y>=z :
            i_memo2 = 1
            i_memo3 = 2
            max_ampli2 = y
            max_ampli3 = z
        else :
            i_memo2 = 2
            i_memo3 = 1
            max_ampli2 = z
            max_ampli3 = y
    if y >= x and y >= z:
        i_memo1 = 1
        max_ampli1 = y
        if x>=z :
            i_memo2 = 0
            i_memo3 = 2
            max_ampli2 = x
            max_ampli3 = z
        else :
            i_memo2 = 2
            i_memo3 = 0
            max_ampli2 = z
            max_ampli3 = x
    if z >= y and z >= x:
        i_memo1 = 2
        max_ampli1 = z
        if y>=x :
            i_memo2 = 1
            i_memo3 = 0
            max_ampli2 = y
            max_ampli3 = x
        else :
            i_memo2 = 0
            i_memo3 = 1
            max_ampli2 = x
            max_ampli3 = y
    
    for a in Ampli[3:] :
        i += 1
        if a >= max_ampli1 :
            max_ampli3 = max_ampli2
            max_ampli2 = max_ampli1
            max_ampli1 = a
            i_memo3 = i_memo2
            i_memo2 = i_memo1
            i_memo1 = i
        else : 
            if a >= max_ampli2 :
                max_ampli3 = max_ampli2
                max_ampli2 = a
                i_memo3 = i_memo2
                i_memo2 = i
            else: 
                if a >= max_ampli3:
                    max_ampli3 = a
                    i_memo3 = i
    return(i_memo1,i_memo2,i_memo3)

def fourier(number,p):
    plt.close()
    T, X, time_memo, delta_x, maxbound, List_Kmax, Parameters_fixed_kernel, Parameters_init = extract_para()
    Parameters_play_kernel,Age,Time,Mass1,Mass2,Time_density,Density1,Density2 = extract_info(number)
    if p==1:
        Mass = Mass1
    else : 
        Mass = Mass2
    X = fft(Mass[len(Mass1)//2:])  # Transformée de fourier
    freq = fftfreq(len(X), d=Time[1]-Time[0])  # Fréquences de la transformée de Fourier
    Y = irfft(X,len(X))
    
    Ampli = abs(X)
    i1,i2,i3 = find3max(Ampli)
    max_domi_freq = max(freq[i1],freq[i2],freq[i3])

    plt.subplot(3,1,1)
    plt.scatter(freq, X.real,s=2, label="Real part")
    plt.scatter(freq, X.imag,s=2, label="Imaginary part")
    plt.axvline(freq[i1],linestyle = '--')
    plt.axvline(freq[i2],linestyle = '--')
    plt.axvline(freq[i3],linestyle = '--')
    plt.grid()
    plt.legend()
    plt.xlabel(r"Frequency (Hz)")
    plt.ylabel(r"Amplitude $X(f)$")
    plt.title("Fourier Transform")
    
    plt.subplot(3,1,2)
    plt.scatter(freq, X.real,s=8, label="Real part")
    plt.scatter(freq, X.imag,s=8, label="Imaginary part")
    plt.axvline(freq[i1],linestyle = '--')
    plt.axvline(freq[i2],linestyle = '--')
    plt.axvline(freq[i3],linestyle = '--')
    plt.xlim(-2*max_domi_freq,2*max_domi_freq)
    plt.grid()
    plt.legend()
    plt.xlabel(r"Frequency (Hz)")
    plt.ylabel(r"Amplitude $X(f)$")
    plt.title("Zoom on the Fourier Transform")
    plt.tight_layout()
    
    plt.subplot(3,1,3)
    plt.plot(Time[len(Mass)//2:], Y, label="Approximated mass%s"%(p))
    plt.plot(Time, Mass, label="Exact mass%s"%(p))
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Mass")
    plt.title("Comparaison between the mass and its estimation for state %s"%(p))
    plt.tight_layout()
    plt.show()
    
    print(1/max_domi_freq)
    
    
    
def colorgraph(number,p,t0,t1):
    # window of observation included in [t0,t1] inter [0;T]
    plt.close()
    T, X, time_memo, delta_x, maxbound, List_Kmax, Parameters_fixed_kernel, Parameters_init = extract_para()
    Parameters_play_kernel,Age,Time,Mass1,Mass2,Time_density,Density1,Density2 = extract_info(number)
    
    delta_t = Time[1]-Time[0]
    I = np.size(Time)
    J = np.size(Age)
    Kmax1 , Kmax2 = List_Kmax
    sl1,sr1,xmin1,xmax1 = Parameters_fixed_kernel[0]
    sl2,sr2,xmin2,xmax2 = Parameters_fixed_kernel[1]
    c1,mu1,sigma1 = Parameters_init[0]
    c2,mu2,sigma2 = Parameters_init[1]
    a1,b1 = Parameters_play_kernel[0]
    a2,b2 = Parameters_play_kernel[1]
    M_tot = Mass1[0]+Mass2[0]
    val_min1 = -Kmax1/M_tot
    val_min2 = -Kmax2/M_tot
    
    if p == 1:
        Density = Density1
    else :
        Density = Density2
    
    i = 0
    while i < len(Time_density) and Time_density[i]<t0:
        i += 1
    j = 0
    while j < len(Time_density) and Time_density[j]<t1:
        j += 1
    
    # setup the normalization and the colormap
    normalize = mcolors.Normalize(vmin=Time_density[i], vmax=Time_density[j-1])
    colormap = cm.jet

    # plot
    for n in range(i,j-1):
        plt.plot(Age,Density[n], color=colormap(normalize(Time_density[n])))
        
    # setup the colorbar
    scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
    scalarmappaple.set_array(Time_density[i:j])
    plt.colorbar(scalarmappaple, ax = None)
    plt.suptitle("Proportion of time spend in state 1 over time t \n T = %s , X = %s , delta_t = %s , delta_x = %s , I = %s , J = %s , \n Kmax1 = %s , sl1 = %s , sr1 = %s , xmin1 = %s , xmax1 = %s \n Kmax2 = %s , sl2 = %s , sr2 = %s , xmin2 = %s , xmax2 = %s , \n c1 = %s , mu1 = %s , sigma1 = %s, c2 = %s , mu2 = %s , sigma2 = %s \n (Minimum value for a1 and b1 = %s) a1 = %s , b1 = %s , (Minimum value for a2 and b2 = %s) a2 = %s , b2 = %s"
              %(T,X,round(delta_t,5),round(delta_x,5),I,J,round(Kmax1,4),round(sl1,4),round(sr1,4),round(xmin1,4),round(xmax1,4),round(Kmax2,4),round(sl2,4),round(sr2,4),round(xmin2,4),round(xmax2,4),round(c1,4),round(mu1,4),round(sigma1,4),round(c2,4),round(mu2,4),round(sigma2,4),round(val_min1,4),round(a1,4),round(b1,4),round(val_min2,4),round(a2,4),round(b2,4)), fontsize=6)
    plt.title("Evolution of the density of state %s from %s to %s"%(p,round(Time_density[i],3),round(Time_density[j-1],3)))
    plt.xlabel("Age")
    plt.ylabel("Density")
    plt.legend()
    # show the figure
    plt.show()

    
def all_anim():
    Sobol = extract_sobol()
    n = np.shape(Sobol)[0]
    for i in range(n):
        animation_density_separated(i)
    return()

def all_mass():
    Sobol = extract_sobol()
    n = np.shape(Sobol)[0]
    for i in range(n):
        graph_mass(i)
    return()

def all_prop():
    Sobol = extract_sobol()
    n = np.shape(Sobol)[0]
    for i in range(n):
        graph_occupation_mean(i)
    return()

def graph_sobol():
    Sobol = extract_sobol()
    X1 , Y1 , X2 , Y2 = [] , [] , [] , []
    for s in Sobol:
        a,b,c,d = s
        X1 += [a]
        Y1 += [b]
        X2 += [c]
        Y2 += [d]
    fig , (ax1,ax2) = plt.subplots(2,1)
    ax1.scatter(X1,Y1,label = "Sobol points for phase 1")
    ax2.scatter(X2,Y2,label = "Sobol points for phase 2")
    plt.plot()
    return()
    
    

def graph_loop(number):
    animation_density_separated(number)
    graph_mass(number)
    print('Done for sim number '+ str(number))
    return()


# if __name__== "__main__":
#     n_proc = 4
#     Sobol = extract_sobol()
#     n_it = np.shape(Sobol)[0]

#     Parallel(n_jobs=n_proc)(delayed(graph_loop)(ie) for ie in range(n_it))
