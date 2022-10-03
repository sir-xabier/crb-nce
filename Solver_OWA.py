import sys, os
from argparse import Namespace
import datetime
import time
import numpy as np
from scipy.optimize import NonlinearConstraint, Bounds, differential_evolution
from tqdm import tqdm
def diferential_evolution(N,orness):
    
    # Función objetivo del problema de optimización
    def objective(w):
        n=w.shape[0]
        obj=0
        for i in range(0,n):
            if w[i] > 0: # Si w[i]=0 la contribución es 0
                obj=obj+w[i]*np.log(w[i])
        return obj

    # Computo de suma para la restricción de que el vector de pesos w sume 1
    def constr_sum_w(w):
        return np.sum(w)

    nlc1 = NonlinearConstraint(constr_sum_w, 1, 1)

    # Calcula el orness de un vector de pesos para la restricción al orness dado
    def constr_orness_w(w):
        n=w.shape[0]
        sum=0
        for i in range(1,n+1):
            sum=sum+(n-i)*w[i-1]
        orness=(1/(n-1))*sum
        return orness

    nlc2 = NonlinearConstraint(constr_orness_w, orness, orness)

    lb=[] # Vector de cotas inferiores para cada elemento del vector de pesos
    ub=[] # Vector de cotas superiores para cada elemento del vector de pesos
    x0=[] # Vector de valores iniciales para cada elemento del vector de pesos
    kf=[] # Vector de booleanos para obligar que los pesos sean factibles

    for i in range(0,N):
        lb.append(0.)
        ub.append(1.)
        if orness > 0.4 and orness < 0.6:
            x0.append(1/N)
        elif orness > 0.2 and orness <= 0.4:
            if i < N/2:
                x0.append(0)
            else:
                x0.append(1/(N/2))
        elif orness <= 0.2:
            if i == N-1:
                x0.append(1)
            else:
                x0.append(0)
        elif orness >= 0.6 and orness < 0.8:
            if i > N/2:
                x0.append(0)
            else:
                x0.append(1/(N/2))
        elif orness >= 0.8:
            if i == 0:
                x0.append(1)
            else:
                x0.append(0)
        kf.append(True)

    x0=x0/np.sum(x0) # Se normaliza el vector de pesos iniciales para que sume 1

    bounds = Bounds(lb, ub, keep_feasible=kf) # Objeto Bounds para DE


    result = differential_evolution(objective, bounds, constraints=(nlc1,nlc2), 
                                    popsize=1, x0=x0, workers=-1, seed=131416, 
                                    updating='deferred')
    return result.x # Devuelve el vector de pesos óptimo


def experiment_diferential_evolution(args):
    N=args.N
    orness=args.orness
    method=args.method
    ROOT= os.path.abspath(os.path.join(os.getcwd(), os.pardir))

    weight_folder = ROOT+"/data/weights/"
    if not os.path.exists(weight_folder):
        os.makedirs(weight_folder)
        
    for i in tqdm(range(0,len(N))):    
        for j in orness:
            file_path=weight_folder+ "W_"+ str(N[i]) + '_' + str(j)
            time_on= time.time()
            if not os.path.exists(file_path):
                w_optim=diferential_evolution(N[i],j)
                np.save(file=file_path,arr=w_optim)
            time_diff= time.time()- time_on    
            print("Experiment " + "W_"+ str(N[i]) + '_' + str(j)+ "has finished in: " + str(time_diff)+ " seconds")     

# Python program to use
# main for function call.
if __name__ == "__main__":        
    # Para cada sesión creamos un directorio nuevo, a partir de la fecha y hora de su ejecución:
    date = datetime.datetime.now()

    #Config
    args = Namespace(
    # Training hyperparameters
    N=[500,1250,10000,150,178,277,17976,214,336,106,297,625,208],
    seed=31416,

    method="differential_evolution",

    orness=[0.1,0.15,0.20,.25,0.3,0.35,0.4,0.45],
    N_particles=12,
    G=20,
    iter_max=300
    )
 
    if args.method=="differential_evolution":
        experiment_diferential_evolution(args)
