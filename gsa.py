import numpy as np
import random as rd
from math import sqrt,exp

def main(f):

    T = 200
    dim = 10
    N = 50
    t = 0
    epsilon = 0.0001
    K = N

    liste_x = np.zeros((N,dim))
    liste_eval = np.zeros(N)
    liste_eval_sorted = np.zeros(N)
    liste_mass = np.zeros(N)
    liste_Mass = np.zeros(N)
    liste_R = np.zeros((N,N))
    liste_force_indiv = np.zeros((N,N,dim))
    liste_force =  np.zeros((N,dim))
    liste_acceleration = np.zeros((N,dim))
    liste_vitesse = np.zeros((N,dim))
    size = np.zeros((dim,2))
    among_K_best = np.ones(N)
    liste_result_x = np.zeros((T,N,dim))
    liste_result_k = np.zeros((T,N))

    search_size(size,dim)
    randomized_ini(liste_x,N,dim,size)

    for t in range(T):
        for i in range(N):
            for d in range(dim):
                liste_result_x[0][i][d] = liste_x[i][d]
            liste_result_k[0][i] = among_K_best[i]

        if K > 1:
            K= int(N * (T-t)/T)
        evaluation(liste_x,f,liste_eval,dim)
        update_masses(liste_x,liste_eval,liste_Mass,liste_mass)
        G = update_graviational_constant(liste_x,t,T)
        find_best_K_indiv(among_K_best,K,liste_eval,liste_eval_sorted,N)
        calculate_forces(liste_x,dim,epsilon,liste_R,liste_Mass,G,liste_force_indiv,liste_force,among_K_best)
        update_acceleration(liste_x,dim,liste_Mass,liste_force,liste_acceleration)
        update_vitesse(liste_x,dim,liste_vitesse,liste_acceleration)
        update_position(liste_x,dim,liste_vitesse)

    max = liste_eval[0]
    for i in range(len(liste_eval)):
        if max < liste_eval[i]:
            max = liste_eval[i]
    return max
    

def search_size(size,dim):
    for d in range(dim):
        size[d][0]=-20
        size[d][1]= 20

    return size

def randomized_ini(liste_x,N,dim,size):
    for i in range(N):
        for d in range(dim):
            liste_x[i][d]=x=rd.random()*(size[d][1]-size[d][0]) + size[d][0]
    return liste_x

def evaluation(liste_x,f,liste_eval,dim):
    for i in range(len(liste_x)):
            liste_eval[i] = f(liste_x[i],dim)

def update_masses(liste_x,liste_eval,liste_Mass,liste_mass):
    best = max(liste_eval)
    worst = min(liste_eval)
    s = 0

    for i in range(len(liste_x)):
        liste_mass[i] = (liste_eval[i]-worst)/(best-worst)
        s+=liste_mass[i]

    for i in range(len(liste_x)):
        liste_Mass[i] = liste_mass[i]/s


def update_graviational_constant(liste_x,t,T):
    G0 = 100
    alpha = 20
    G = G0 * exp(-alpha * t / T)
    return G

def find_best_K_indiv(among_K_best,K,liste_eval,liste_eval_sorted,N):
    for i in range(len(among_K_best)):
        among_K_best[i] = 0
        liste_eval_sorted[i] = liste_eval[i]
    liste_eval_sorted.sort()
    lowerbound = liste_eval_sorted[N-K]


    for i in range(len(among_K_best)):
        if liste_eval[i] >= lowerbound:
            among_K_best[i]=1

def calculate_forces(liste_x,dim,epsilon,liste_R,liste_Mass,G,liste_force_indiv,liste_force,among_K_best):
    for i in range(len(liste_x)):
        for j in range(i+1,len(liste_x)):
            s = 0
            for d in range(dim):
                s+= (liste_x[j][d]-liste_x[i][d])**2
            liste_R[i][j] = sqrt(s)
            liste_R[j][i] = sqrt(s)

    for i in range(len(liste_x)):
        for d in range(dim):
            liste_force[i][d] = 0
        for j in range(len(liste_x)):
            if i==j:
                continue
            for d in range(dim):
                liste_force_indiv[i][j][d] = G * liste_Mass[i] * liste_Mass[j] * (liste_x[j][d]-liste_x[i][d])/(liste_R[i][j] + epsilon)
                if among_K_best[j]==1:
                    liste_force[i][d] += rd.random()*liste_force_indiv[i][j][d]

    
def update_acceleration(liste_x,dim,liste_Mass,liste_force,liste_acceleration):
    for i in range(len(liste_x)):
        for d in range(dim):
            if liste_Mass[i] == 0:
                liste_acceleration[i][d] = 0
            else:
                liste_acceleration[i][d] = liste_force[i][d]/liste_Mass[i]

def update_vitesse(liste_x,dim,liste_vitesse,liste_acceleration):
    for i in range(len(liste_x)):
        for d in range(dim):
            liste_vitesse[i][d] = rd.random()*liste_vitesse[i][d] + liste_acceleration[i][d]

def update_position(liste_x,dim,liste_vitesse):
    for i in range(len(liste_x)):
        for d in range(dim):
            liste_x[i][d] = liste_x[i][d] + liste_vitesse[i][d]


    
def f(listx,dim):
    s = 0
    for d in range(dim):
        s+= listx[d]**2
    return -s+400

print(main(f))