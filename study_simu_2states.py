#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 10:15:05 2023

@author: ldumetz

Numerical exploration for the 2 phases circadian clock system
Exploit already created files taht contains the usel data
(densities, value sof parameters,...)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
# from joblib import Parallel, delayed
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
        
    final_time, final_age, time_memo, delta_x, maxbound = Forstep
    
    #______________________________________________________________
    
    fichier_kmax = open("List_kmax_.txt", "r")
    list_kmax = []
    for l in fichier_kmax.readlines():
        list_kmax += [float(l)]
        
    #______________________________________________________________   
        
    fichier_para_fixed = open("Parameters_fixed_kernel_it_.txt", "r")
    l1_para_fixed , l2_para_fixed = fichier_para_fixed.readlines()
    parameters_fixed_kernel = []
    L1_fixed , L2_fixed = [] , []
    for s in l1_para_fixed.split() :
        L1_fixed += [float(s)]
    for s in l2_para_fixed.split() :
        L2_fixed += [float(s)]
    parameters_fixed_kernel += [L1_fixed, L2_fixed]
    
    #______________________________________________________________   
        
    fichier_para_init = open("Parameters_initial_it_.txt", "r")
    l1_para_init , l2_para_init = fichier_para_init.readlines()
    parameters_init = []
    L1_init , L2_init = [] , []
    for s in l1_para_init.split() :
        L1_init += [float(s)]
    for s in l2_para_init.split() :
        L2_init += [float(s)]
    parameters_init += [L1_init, L2_init]
    
    
    return(final_time, final_age, time_memo, delta_x, maxbound, 
           list_kmax, parameters_fixed_kernel, parameters_init)


def extract_info(number):
    fichier_data = open("parameters_play_it_%s.txt"%(number), "r")
    l1 , l2 = fichier_data.readlines()
    para1 , para2 = [] , []
    for s in l1.split():
        para1 += [float(s)]
    for s in l2.split():
        para2 += [float(s)]
    parameters_play_kernel = [para1, para2]
    
    #______________________________________________________________  
    
    fichier_age = open("age_mass_it_%s.txt"%(number), "r")
    age = []
    for l in fichier_age.readlines():
        age += [float(l)]
    
    #______________________________________________________________  
    
    fichier_density1 = open("density_1_it_%s.txt"%(number), "r")
    density1 = []
    for l in fichier_density1.readlines():
        L = []
        for s in l.split():
            L += [float(s)]
        density1 += [L]
        
    #______________________________________________________________  
     
    fichier_density2 = open("density_2_it_%s.txt"%(number), "r")
    density2 = []
    for l in fichier_density2.readlines():
        L = []
        for s in l.split():
             L += [float(s)]
        density2 += [L]
        
    #______________________________________________________________  
    
    fichier_mass1 = open("mass_1_it_%s.txt"%(number), "r")
    mass1 = []
    for l in fichier_mass1.readlines():
        mass1 += [float(l)]
    
    #______________________________________________________________  
    
    fichier_mass2 = open("mass_2_it_%s.txt"%(number), "r")
    mass2 = []
    for l in fichier_mass2.readlines():
        mass2 += [float(l)]
        
    #______________________________________________________________  
    
    fichier_time_density = open("time_density_it_%s.txt"%(number), "r")
    time_density = []
    for l in fichier_time_density.readlines():
        time_density += [float(l)]
        
    #______________________________________________________________  
    
    fichier_time = open("time_mass_it_%s.txt"%(number), "r")
    time = []
    for l in fichier_time.readlines():
        time += [float(l)]
    
    return(parameters_play_kernel, age, time, mass1, mass2,
           time_density, density1, density2)


def kernel_p(t, x , p, mass1, mass2, list_kmax, 
             parameters_fixed_kernel, parameters_play_kernel): 
    kmaxp = list_kmax[p-1]
    sl, sr, xmin, xmax = parameters_fixed_kernel[p - 1]
    a , b = parameters_play_kernel[p - 1]
    if p == 1:
        mass_self , mass_other = mass1 , mass2
    else :
        mass_self , mass_other = mass2 , mass1
    coeff = a * mass_other + b * mass_self + kmaxp
    result = coeff * (np.tanh(sl * (x - xmin)) + np.tanh(sr * (xmax - x))) / 2
    return(result)



def animation_density_separated(number):
    final_time, final_age, time_memo, delta_x, maxbound, list_kmax, parameters_fixed_kernel, parameters_init = extract_para()
    parameters_play_kernel, age, time, mass1, mass2, time_density, density1, density2 = extract_info(number)
    # delta_t = time[1] - time[0]
    # I = np.size(time)
    # J = np.size(age)
    kmax1, kmax2 = list_kmax
    sl1, sr1, xmin1, xmax1 = parameters_fixed_kernel[0]
    sl2, sr2, xmin2, xmax2 = parameters_fixed_kernel[1]
    c1, mu1, sigma1 = parameters_init[0]
    c2, mu2, sigma2 = parameters_init[1]
    a1, b1 = parameters_play_kernel[0]
    a2, b2 = parameters_play_kernel[1]
    # m_tot = mass1[0] + mass2[0]
    # val_min1 = -kmax1/m_tot
    # val_min2 = -kmax2/m_tot
    
    
    ecart = 0
    while time[ecart] <= time_memo:
        ecart += 1
    
    memory_K1 = [[kernel_p(time[0], x, 1, mass1[0], mass2[0], list_kmax,
                parameters_fixed_kernel, parameters_play_kernel) for x in age]]
    memory_K2 = [[kernel_p(time[0], x, 2, mass1[0], mass2[0], list_kmax,
                parameters_fixed_kernel, parameters_play_kernel) for x in age]]
    
    compt = 0
    
    for i in range(len(time)-1):
        ts = time[i+1]
        if compt == ecart:
            compt = 0
            memory_K1 += [[kernel_p(ts, x, 1, mass1[i + 1], mass2[i + 1],
                        list_kmax, parameters_fixed_kernel, parameters_play_kernel) for x in age]]
            memory_K2 += [[kernel_p(ts, x, 2, mass1[i + 1], mass2[i + 1],
                        list_kmax, parameters_fixed_kernel, parameters_play_kernel) for x in age]]
        else :
            compt += 1
    
    
    fig, (ax0,ax1) = plt.subplots(2,1,figsize=(12,10))
    ax0.set_ylim(0, 1.2)
    ax0.set_xlim(0, 1.01*final_age)
    ax0.set_xlabel("Age")
    ax0.set_ylabel("Value")
    ax0.grid()

    
    ax1.set_ylim(0, 1.2)
    ax1.set_xlim(0, 1.01*final_age)
    ax1.set_xlabel("Age")
    ax1.set_ylabel("Value")
    ax1.grid()

            
    plt.suptitle("Evolution of the density and the kernel over time t", fontsize=10)
    

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
        t = time_density[i]
        plt.title("Current time =%s"%(round(t,2)),fontsize = 10)
        densityani1.set_data(age, density1[i])
        densityani2.set_data(age, density2[i])
        kernelani1.set_data(age, memory_K1[i])
        kernelani2.set_data(age,memory_K2[i])
        ax0.legend(loc = 'upper left')
        ax1.legend(loc = 'upper left')
        return (densityani1,densityani2,kernelani1,kernelani2)

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                             frames=len(time_density), interval=100, blit=True,
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
    plt.rcParams.update({'font.size': 16})
    
    final_time, final_age, time_memo, delta_x, maxbound, list_kmax, parameters_fixed_kernel, parameters_init = extract_para()
    parameters_play_kernel, age, time, mass1, mass2, time_density, density1, density2 = extract_info(number)
    # delta_t = time[1] - time[0]
    # I = np.size(time)
    # J = np.size(age)
    kmax1, kmax2 = list_kmax
    sl1, sr1, xmin1, xmax1 = parameters_fixed_kernel[0]
    sl2, sr2, xmin2, xmax2 = parameters_fixed_kernel[1]
    c1, mu1, sigma1 = parameters_init[0]
    c2, mu2, sigma2 = parameters_init[1]
    a1, b1 = parameters_play_kernel[0]
    a2, b2 = parameters_play_kernel[1]
    M_tot = mass1[0] + mass2[0]
    # val_min1 = -kmax1/M_tot
    # val_min2 = -kmax2/M_tot
    
    mass_tot = []
    for i in range(len(mass1)):
        m = mass1[i] + mass2[i]
        mass_tot += [m]
    
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
    
    print("Evolution of the mass over time t")
    
    plt.gcf().subplots_adjust(wspace = 0, hspace = 0.5)
    plt.suptitle("Evolution of the masses over time", fontsize = 25)
    ax0.set_title("Mass of state 1")
    ax0.set_ylim(0, 1.01*M_tot) 
    ax0.set_xticks(np.arange(0, final_time, 10))
    ax0.set_yticks(np.arange(0, 1.01 * M_tot, 0.5))
    ax0.plot(time, mass1, label = "State 1")
    ax1.set_title("Mass of state 2")
    ax1.set_ylim(0, 1.01 * M_tot) 
    ax1.set_xticks(np.arange(0, final_time, 10))
    ax1.set_yticks(np.arange(0, 1.01 * M_tot, 0.5))
    ax1.plot(time, mass2, label = "State 2")
    axt.set_title("Total mass")
    axt.set_ylim(0, 1.01 * M_tot)
    axt.set_xticks(np.arange(0, final_time, 10))
    axt.set_yticks(np.arange(0, 1.01 * M_tot, 0.5))
    axt.plot(time, mass_tot, label = "Total")
    
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
    final_time, final_age, time_memo, delta_x, maxbound, list_kmax, parameters_fixed_kernel, parameters_init = extract_para()
    parameters_play_kernel, age, time, mass1, mass2, time_density, density1, density2 = extract_info(number)
    # delta_t = time[1] - time[0]
    # I = np.size(time)
    # J = np.size(age)
    kmax1, kmax2 = list_kmax
    sl1, sr1, xmin1, xmax1 = parameters_fixed_kernel[0]
    sl2, sr2, xmin2, xmax2 = parameters_fixed_kernel[1]
    c1, mu1, sigma1 = parameters_init[0]
    c2, mu2, sigma2 = parameters_init[1]
    a1, b1 = parameters_play_kernel[0]
    a2, b2 = parameters_play_kernel[1]
    # M_tot = mass1[0] + mass2[0]
    # val_min1 = -kmax1/M_tot
    # val_min2 = -kmax2/M_tot
    
    
    S = 0
    X = []
    Y = []
    for i in range(len(time)):
        if mass1[i] >= mass2[i]:
            S += 1
        X += [time[i]]
        Y += [S/(i+1)]
    
    plt.plot(X,Y)
    
    plt.suptitle("Proportion of time spend in state 1 over time t", fontsize=6)
    
    
    
    plt.savefig('Graph_prop_%s.png'%(number))
    return()

    
def find3max(Ampli):
    """
    Ampli is a list of amplitude associated to Fourier frequencies
    We want to find the indices at which the 3 largests amplitudes occur
    """
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
    return(i_memo1, i_memo2, i_memo3)

def fourier(number,p):
    plt.close()
    final_time, final_age, time_memo, delta_x, maxbound, list_kmax, parameters_fixed_kernel, parameters_init = extract_para()
    parameters_play_kernel, age, time, mass1, mass2, time_density, density1, density2 = extract_info(number)
    if p==1:
        Mass = mass1
    else : 
        Mass = mass2
    X = fft(Mass[len(mass1)//2:])  # Fourier transform
    freq = fftfreq(len(final_age), d=time[1] - time[0])  # Frequencies
    Y = irfft(X,len(X)) # Graph of the approximated curve
    
    Ampli = abs(X)
    i1,i2,i3 = find3max(Ampli)
    max_domi_freq = max(freq[i1], freq[i2], freq[i3])

    plt.subplot(3, 1, 1)
    plt.scatter(freq, X.real, s=2, label="Real part")
    plt.scatter(freq, X.imag, s=2, label="Imaginary part")
    plt.axvline(freq[i1], linestyle = '--')
    plt.axvline(freq[i2], linestyle = '--')
    plt.axvline(freq[i3], linestyle = '--')
    plt.grid()
    plt.legend()
    plt.xlabel(r"Frequency (Hz)")
    plt.ylabel(r"Amplitude $X(f)$")
    plt.title("Fourier Transform")
    
    plt.subplot(3, 1, 2)
    plt.scatter(freq, X.real, s=8, label="Real part")
    plt.scatter(freq, X.imag, s=8, label="Imaginary part")
    plt.axvline(freq[i1], linestyle = '--')
    plt.axvline(freq[i2], linestyle = '--')
    plt.axvline(freq[i3], linestyle = '--')
    plt.xlim(-2 * max_domi_freq, 2 * max_domi_freq)
    plt.grid()
    plt.legend()
    plt.xlabel(r"Frequency (Hz)")
    plt.ylabel(r"Amplitude $X(f)$")
    plt.title("Zoom on the Fourier Transform")
    plt.tight_layout()
    
    plt.subplot(3, 1, 3)
    plt.plot(time[len(Mass)//2 :], Y, label="Approximated mass%s"%(p))
    plt.plot(time, Mass, label="Exact mass %s"%(p))
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Mass")
    plt.title("Comparaison between the mass and its estimation for state %s"%(p))
    plt.tight_layout()
    plt.show()
    
    print(1/max_domi_freq)
    
    
    
def colorgraph(number,p,t0,t1):
    # window of observation included in [t0,t1] inter [0;T]
    
    # for the fontsize
    plt.rcParams.update({'font.size': 20})
    
    plt.close()
    final_time, final_age, time_memo, delta_x, maxbound, list_kmax, parameters_fixed_kernel, parameters_init = extract_para()
    parameters_play_kernel, age, time, mass1, mass2, time_density, density1, density2 = extract_info(number)
    
    # delta_t = time[1] - time[0]
    # I = np.size(time)
    # J = np.size(age)
    kmax1, kmax2 = list_kmax
    sl1, sr1, xmin1, xmax1 = parameters_fixed_kernel[0]
    sl2, sr2, xmin2, xmax2 = parameters_fixed_kernel[1]
    c1, mu1, sigma1 = parameters_init[0]
    c2, mu2, sigma2 = parameters_init[1]
    a1, b1 = parameters_play_kernel[0]
    a2, b2 = parameters_play_kernel[1]
    # M_tot = mass1[0] + mass2[0]
    # val_min1 = -kmax1/M_tot
    # val_min2 = -kmax2/M_tot
    
    # to extract the good mass to show
    ecart_memo_mass = 0
    while time[ecart_memo_mass] <= time_memo:
        ecart_memo_mass += 1
    compt_mass = 0
    mass_extract_1 = [mass1[0]]
    mass_extract_2 = [mass2[0]]
    for i in range(len(time)-1): 
        if compt_mass == ecart_memo_mass:
            compt_mass = 0
            mass_extract_1 += [mass1[i]]
            mass_extract_2 += [mass2[i]]
        else :
            compt_mass += 1
    
    if p == 1:
        density = density1
    else :
        density = density2
    
    i = 0
    while i < len(time_density) and time_density[i]<t0:
        i += 1
    j = 0
    while j < len(time_density) and time_density[j]<t1:
        j += 1
    
    # we want around 10 graphs on the picture
    ecart = int((j-i) / 10)
    list_extract_density_p = []
    list_extract_kernel = []
    current_time = []
    
    compt = ecart
    for k in range(i, j) : 
        if compt == ecart:
            compt = 0
            list_extract_density_p += [density[k]]
            list_extract_kernel += [[kernel_p(time_density[k], x, p, mass_extract_1[k], mass_extract_2[k],
                                    list_kmax, parameters_fixed_kernel, parameters_play_kernel) for x in age]]
            current_time += [time_density[k]]
        else :
            compt += 1
    
    

    
    # setup the normalization and the colormap
    normalize = mcolors.Normalize(vmin = current_time[0], 
                                  vmax = current_time[-1])
    colormap1 = cm.jet
    colormap2 = cm.binary
    
    fig, ax = plt.subplots()
    for n in range(len(current_time)):
        ax.plot(age, list_extract_density_p[n], 
                 color=colormap1(normalize(current_time[n])))
        ax.plot(age, list_extract_kernel[n], 
                 color=colormap2(normalize(current_time[n])))
        
        
    # setup the colorbar
    scalarmappaple1 = cm.ScalarMappable(norm=normalize, cmap=colormap1)
    scalarmappaple1.set_array(current_time)
    scalarmappaple2 = cm.ScalarMappable(norm=normalize, cmap=colormap2)
    scalarmappaple2.set_array(current_time)
    plt.colorbar(scalarmappaple1, ax = None).set_label(label='Density',size=24)
    plt.colorbar(scalarmappaple2, ax = None).set_label(label='Kernel',size=24)
    plt.title("Evolution of the density of state %s from %s to %s"%(p+1, round(current_time[0], 2),
                                                    round(current_time[-1],0)),fontsize=30)
    plt.xlabel("Age",fontsize=25)
    plt.ylabel("Density",fontsize=25)
    plt.legend()
    plt.show()
    # print("Delta t = %s , a1 = %s , b1 = %s , a2 = %s , b2 = %s"%(delta_t,a1,b1,a2,b2))

    
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