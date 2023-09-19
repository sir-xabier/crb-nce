import sys, os
from argparse import Namespace
import time
import numpy as np
from scipy.linalg import null_space

sys.path.append(os.getcwd())

def fobj(w):
    n=w.shape[0]
    obj=0
    for i in range(0,n):
        if w[i] > 0: # Si w[i]=0 la contribución es 0
            obj=obj+w[i]*np.log(w[i])
    return obj

gradf=lambda x: 1+np.log(x)
norm=lambda x: np.sqrt(np.inner(x,x))

def ev_orness(w):
    n=w.shape[0]
    suma=0
    for i in range(1,n+1):
        suma=suma+(n-i)*w[i-1]
    orness=(1/(n-1))*suma
    return orness

def desc_grad(x0,P,ls=1,itermax=130000,tol=1e-15,disp=True):
    status=0
    f0=fobj(x0)
    x_ant=None
    mod_ls=False
    kd=0
    dif=np.ones(10)
    km=0
    for i in range(1,itermax+1):
        if disp:
            h=''
            kd+=1
        g=-1*gradf(x0)
        g=g/norm(g)
        g=np.matmul(P,g)
        norm_g=norm(g)
        if norm_g < tol:
            print("Convergencia 1. Paso = " + str(ls) + "  Gradiente = " + str(norm_g))
            status=3
        if status==0:
            g=g/norm_g
            B=np.array([-x0/g,(1-x0)/g])
            mi=np.amax(np.amin(B,axis=0))
            ma=np.amin(np.amax(B,axis=0))
            if mi < ma and ma > 0:
                if min(ma,ls)==ma:
#                    print("a")
                    if disp: h+='a'
                    l1=ma/4
                    l2=3*ma/4
                    l=ma/2
                    ls=ls*0.9
                else:
                    if disp: h+='b'
  #                  print("b")
                    l1=ls/2
                    l2=(ls+ma)/2
                    l=ls
                    mod_ls=True
                    
                x1=x0+l1*g
                x2=x0+l2*g
                x=x0+l*g
                f1=fobj(x1)
                f2=fobj(x2)
                f=fobj(x)
                
                x0_n=x
                f0_n=f
                if np.argmin([f1,f,f2]) == 0:
                    if disp: h+='1'
                    x0_n=x1
                    f0_n=f1
                    if mod_ls:
                        mod_ls=False
                        ls=ls*0.75
                elif np.argmin([f1,f,f2]) == 2:
                    if disp: h+='2'
                    x0_n=x2
                    f0_n=f2
                    if mod_ls:
                        mod_ls=False
                        ls=ls*1.25
            else:
                print("Problema gradiente apunta infactible")
                status=1
            
            if i >= 2 and x_ant is not None and status==0:
                g=x0_n - x_ant 
                B=np.array([-x_ant/g,(1-x_ant)/g])
                mi=np.amax(np.amin(B,axis=0))
                ma=np.amin(np.amax(B,axis=0))
                if mi < ma and ma > 1:
                    x=x_ant+((ma+1)/2)*g
                    f=fobj(x)
                    if f <= f0_n:
                        x0_n=x
                        f0_n=f
                        if disp: h+='c'
                     
            if f0_n < f0 and (x_ant!=x0_n).any() and status==0:
                km+=1
                dif[km-1]=f0 - f0_n
                cambio=norm(x0-x0_n)
                x_ant=x0
                x0=x0_n
                f0=f0_n
                if km == 10: km=0
                
    
            elif status==0:
                if disp: print("Reduce")
                ls=ls/2
            
            if disp and status==0:
                print(h + " " + "F. obj. it. "+ str(i) +": " + str(f0) + " " + str(dif))
                if kd > 500:
                    print("Paso = " + str(ls) + "  Gradiente = " + str(norm_g) + " Dif = " + str(dif) + " Cambio = " + str(cambio))
                    if kd == 550: kd=0
                
            if (np.mean(dif) < tol or ls < tol) and status==0:
                print("Convergencia 2. Paso = " + str(ls) + " Gradiente = " + str(norm_g) + " Dif = " + str(dif) + " Cambio = " + str(cambio))
                status=2

        if status:
            if status==1: 
                break
            elif status==2:
                break
            elif status==3:
                break
    if status==0:
        print("Max iteraciones. Paso = " + str(ls) + "  Gradiente = " + str(norm_g) + " Dif = " + str(dif) + " Cambio = " + str(cambio))
    
    return x0,f0,status

def experiment_DG(args):
    N=args.N
    orness=args.orness
    iter_max=args.iter_max
    tol=args.tol
    weight_folder = os.getcwd()+"/data/weights/DG"
    if not os.path.exists(weight_folder):
        os.makedirs(weight_folder)
        
    for i in N:
        sample_folder = os.path.join(weight_folder, str(i))
        if not os.path.exists(sample_folder):
            os.makedirs(sample_folder)
            
        v1=np.ones(i)
        v2=np.array([i-l for l in range(1,i+1)])
        M=null_space([v1,v2])
        P=np.matmul(M,M.T)
    
        for j in orness:
            file_path=sample_folder+"/"+ "W_"+ str(i) + '_' + str(j)
            time_on= time.time()
            
            k=i-int(j*i)
            alfa=(2*(i-1)*j+k-i)/((k-1)*i)
            beta=(1-(k-1)*alfa)/(i-k+1)
            x0=np.zeros(i)
             
            for l in range(1,i+1):
                if l < k:
                    x0[l-1]=alfa
                else:
                    x0[l-1]=beta
            if abs(ev_orness(x0) - j) < 1e-13 and abs(np.sum(x0)-1) < 1e-13:
                print("Experiment " + "W_"+ str(i) + '_' + str(j) + "has started.")
                if not os.path.exists(file_path):
                    w_optim,f,s=desc_grad(x0,P,itermax=iter_max,tol=tol,disp=False)
                    if s > 1 and abs(ev_orness(w_optim) - j) < 1e-13 and abs(np.sum(w_optim)-1) < 1e-13 and min(w_optim) >= 0 and max(w_optim) <= 1:
                        np.save(file=file_path,arr=w_optim)
                        print("Archivo guardado!!")
                time_diff= time.time()- time_on    
                print("Experiment " + "W_"+ str(i) + '_' + str(j)+ "has finished in: " + str(time_diff)+ " seconds")     
            else:
                print("ERROR: la solución inicial es infactible.")
# Python program to use
# main for function call.
if __name__ == "__main__":        
    # Para cada sesión creamos un directorio nuevo, a partir de la fecha y hora de su ejecución:

    #Config
    args = Namespace(
    # Training hyperparameters
    N=[208],
    #N=[500,1250,150,1797,178,569,106,625,297,336,214,10000],
    tol=1e-14,
    orness=[0.1,0.2,0.3,0.35,0.4,0.45],
    iter_max=100000
    )
     
    experiment_DG(args)
    
