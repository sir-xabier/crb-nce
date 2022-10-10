# -*- coding: utf-8 -*-
from scipy.spatial import distance_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import os

PATH_ROOT=os.path.join('.')

# Calcula la matriz de cubrimientos dado un conjunto de datos y unos centroides
# Parámetros: a controla el grado de cubrimiento a distancia 1
# distance_normalizer es un parámetro de escala para llevar las distancias a 
# [0,1] aprox.
def coverings(X,centroids,a=2*np.log(10),distance_normalizer=1/np.sqrt(2)):
    if len(centroids.shape) == 1:
        centroids=centroids.reshape(1,centroids.shape[0])
    return np.exp(-1*a*distance_normalizer*distance_matrix(X, centroids))

# Dada una matriz de cubrimientos U, devuelve un vector con el cubrimiento
# de cada objeto a su cluster. Con mode=1 cada objeto se asigna al cluster 
# con mayor cubrimiento. Con mode=0 es necesario aportar el vector y con 
# el cluster al que pertenece cada objeto.     
def covering_i(U,mode=1,y=None):
    if mode:
        return np.amax(U,axis=1)
    else:
        cover_i=np.zeros(U.shape[0])
        for i in range(0,U.shape[0]):
            cover_i[i]=U[i,y[i]]
        return cover_i

# Calcula la agregación OWA del vector x usando el vector de pesos w
def OWA(x,w):
    xs=-np.sort(-x)
    return np.dot(xs,w)

# Calcula el índice de cubrimiento global a partir de una matriz de cubrimientos U. 
# Por defecto (mode=1) se calcula como el mínimo por filas de los máximos por columna de U
# Admite cambiar el mínimo por el percentil perc, por OWA de orness dado.
# Para mode=0 U es el vector de cubrimientos de cada objeto en su cluster
# Nótese que al calcular OWA se invoca a la función pesos (no es óptimo)
def global_covering_index(U,function='min',orness=0.5,perc=5,mode=1):
    if mode:
        if function=='percentile':
            return np.percentile(np.amax(U,axis=1),q=perc)
        elif function=='mean':
            return np.mean(np.amax(U,axis=1))
        elif function=='OWA':
            path=PATH_ROOT+"reports/weights/"+ "W_"+ str(U.shape[0]) + '_' + str(orness)
            return OWA(np.amax(U,axis=1),np.load(allow_pickle=True,file=path))
        else:
            return np.amin(np.amax(U,axis=1))
    else:
        if function=='percentile':
            return np.percentile(U,q=perc)
        elif function=='mean':
            return np.mean(U)
        elif function=='OWA':
            path=PATH_ROOT+"reports/weights/"+ "W_"+ str(U.shape[0]) + '_' + str(orness)
            return OWA(U,np.load(allow_pickle=True,file=path))
        else:
            return np.amin(U)

# Calcula centroides a partir de una matriz de datos X y un vector de asignaciones
# a cluster y
def centroids(X,y):
    centroides=np.mean(X[y==0],axis=0)
    for clust in range(1,np.amax(y)+1):
        centroides=np.vstack((centroides,np.mean(X[y==clust],axis=0)))
    return centroides

# Función para graficar scatters de una matriz de datos bidimensionales X
# Coloreando clusters según y
# Pinta los centroides centroids
# Admite incluir un título en el gráfico, por defecto no lo hace (title_bool=0)
# Admite incluir texto en el gráfico, por defecto no lo hace (text_bool=0)
def subplot(nx,ny,ns,X,y,centroids,title_bool=0,title='',
            text_bool=0,text='',text_pos1=0,text_pos2=0,size=15,halign='',
            xlim1=0,xlim2=1,ylim1=0,ylim2=1,xticks=(),yticks=()):
    plt.subplot(nx,ny,ns)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='r' ,marker='+')
    if title_bool:
        plt.title(title)
    plt.xlim(xlim1, xlim2)
    plt.ylim(ylim1, ylim2)
    plt.xticks(xticks)
    plt.yticks(yticks)
    if text_bool:
        plt.text(text_pos1,text_pos2,(text),
                 transform=plt.gca().transAxes,size=size,
                 horizontalalignment=halign,
                 )


# Función para graficar perfiles de GCI (gc) según número de clusters (rango)
# Admite incluir un título en el gráfico, por defecto no lo hace (title_bool=0)
def subplot_codo(nx,ny,ns,rango,gc,title_bool=0,title='',
            xlim1=0,xlim2=1,ylim1=0,ylim2=1,xticks=()):
    plt.subplot(nx,ny,ns)
    if title_bool:
        plt.title(title)  
    plt.plot(rango, gc, color='darkorange', lw=2, marker='o', markersize=6)  
    plt.xticks(xticks)  
    plt.xlim(xlim1, xlim2)  
    plt.ylim(ylim1, ylim2)  
    plt.xlabel('Valores de K')  
    plt.ylabel('Valores de GCI')  

# Modificación de silhouette para que cuando solo haya un cluster en y devuelva None    
def silhouette_score2(X,y):
    if np.amax(y)==0:
        return None
    else:
        return silhouette_score(X,y)

# Modificación de CH para que cuando solo haya un cluster en y devuelva None       
def calinski_harabasz_score2(X,y):
    if np.amax(y)==0:
        return None
    else:
        return calinski_harabasz_score(X,y)

# Modificación de DB para que cuando solo haya un cluster en y devuelva None   
def davies_bouldin_score2(X,y):
    if np.amax(y)==0:
        return None
    else:
        return davies_bouldin_score(X,y)

            
#####################################################################
# CRITERIOS DE SELECCIÓN DE NÚMERO DE CLUSTERS A PARTIR DE PERFIL GCI

# A partir de una métrica x (un valor para cada número de clusters k) devuelve
# el k anterior al primer k (partiendo de un valor mínimo nmin) 
# en que la métrica está por debajo de umbral
def parada_gci(x,nmin,umbral):
    bool=1
    k=nmin
    while bool and k < x.shape[0] + nmin:
        if x[k-nmin] < umbral:
            bool=0
        if bool and k+1 < x.shape[0] + nmin:
            k=k+1
    return k-1

# Criterio sobre diferencia de GCI y ratio de diferencias sucesivas
# Un umbral para la diferencia y otro para el ratio
# Devuelve el último k con ratio y diferencia mayor que el umbral
# nmin sirve para especificar el número inicial de clusters (típicamente 1)
# que se devuelve si no se superan nunca los umbrales     
def parada_gci2(gci,nmin,umbral_diff,umbral_ratio_diff):
    k=nmin
    diff=np.diff(gci)
    for i in range(1,diff.shape[0]):
        ratio_diff=diff[i-1]/diff[i]
        if ratio_diff > umbral_ratio_diff and diff[i-1] > umbral_diff:
            k=i+1
    return k

# Similar al anterior pero considerando también la segunda diferencia de GCI
# así como el incremento relativo de la diferencia (ratio)
# 3 umbrales: para dif, para ratio, y para el ratio de diferencias
# Como la 1ª dif debería ir decreciendo, se controla cuando la 2ª dif 
# pasa a ser > 0 (bool_neg)
# Se asigna una puntuación a cada número de clusters. El número mínimo (suele ser 1)
# recibe 0.9 puntos. 
# Si para un k ratio_diff supera su umbral con dif2 negativa (y la anterior también)
# ese k gana un punto (se suma i/tamaño de diff para que en caso de empate se
# seleccione el k mayor).
# Si además se superan los umbrales para ratio y diff se gana otro punto.
# Se devuelve el k con mayor puntuación
def parada_gci_pts(gci,nmin,umbral_diff,umbral_ratio,umbral_ratio_diff):
    diff=np.diff(gci)
    diff2=np.diff(diff)
    bool_neg=1
    pts=np.zeros(gci.shape[0])
    pts[0]=0.9
    for i in range(1,diff.shape[0]):
        ratio_diff=diff[i-1]/diff[i]
        ratio=(gci[i]-gci[i-1])/gci[i-1]
        if ratio_diff > umbral_ratio_diff and bool_neg and diff2[i-1] < 0:
            pts[i]=1+i/diff.shape[0]
            if diff[i-1] > umbral_diff or ratio > umbral_ratio:
                pts[i]=pts[i]+1
        if diff2[i-1] > 0:
            bool_neg=0
        else:
            bool_neg=1
    return np.argmax(pts)+nmin

# Una variación del anterior. Se controla también si las 2as diferencias 
# a partir de cada k están por debajo de un valor mínimo (eps).
# Ahora k=1 recibe de entrada casi 4 ptos.
# Cada k gana una puntuación creciente entre 0 y 1 (i/diff.shape[0]) para deshacer empates
# en favor de los valores mayores.
# Se da más peso (puntos) a ciertas condiciones que a otras. 
def parada_all(gci,nmin,eps,umbral_ratio_diff,umbral_diff,umbral_ratio):
    diff=np.diff(gci)
    diff2=np.diff(diff)
    pts=np.zeros(gci.shape[0])
    pts[0]=3.999999
    for i in range(1,diff.shape[0]):
        pts[i]=i/diff.shape[0]
        ratio_diff=diff[i-1]/diff[i]
        ratio=(gci[i]-gci[i-1])/gci[i-1]
        if diff[i] > eps:
            if i < diff2.shape[0]-1 and (np.abs(diff2[i:])<eps).all():
                pts[i]=pts[i]+3
            if ratio_diff > umbral_ratio_diff:
                pts[i]=pts[i]+3
            if diff[i-1] > umbral_diff or ratio > umbral_ratio:
                pts[i]=pts[i]+1
    return np.argmax(pts)+nmin

# Todas las condiciones anteriores de manera conjunta
def parada_all2(gci,nmin,eps,umbral_ratio_diff,umbral_diff,umbral_ratio,
                umbral_diff2):
    diff=np.diff(gci)
    diff2=np.diff(diff)
    k=nmin
    for i in range(1,diff.shape[0]):
        ratio_diff=diff[i-1]/diff[i]
        ratio=(gci[i]-gci[i-1])/gci[i-1]
        if i < diff2.shape[0]-1 and (np.abs(diff2[i:])<eps).all() and ratio_diff > umbral_ratio_diff and (diff[i-1] > umbral_diff or ratio > umbral_ratio) and diff2[i-1] < umbral_diff2:
            k=i+1
    return k

# Criterio que auna 10 condiciones, dando 1 pto por condición.
# k=1 recibe 7 ptos, así que un k>1 solo puede ser seleccionado si cumple
# al menos 7 condiciones (cada k>1 recibe una fracción creciente de punto para
# favorecer k's mayores). 
def parada_conds(gci,nmin):
    sin_cubrir=1-gci #proporción sin cubrimiento total
    diff=np.diff(gci)
    diff2=np.diff(diff)
    prop_exp=diff/sin_cubrir[:-1] #proporciones que se cubren en cada k
    pts=np.zeros(gci.shape[0])
    pts[0]=7 #1a
    for i in range(1,diff.shape[0]):
        pts[i]=i/diff.shape[0] #fracción creciente
        ratio_diff=diff[i-1]/diff[i] #condición sobre ratio de diferencias
        if ratio_diff > 2: #1b
            pts[i]=pts[i]+1 #2a
        ratio=(gci[i]-gci[i-1])/gci[i-1] #ratio relativo
        if ratio > 0.04: #2b
            pts[i]=pts[i]+1
        if diff[i-1]>0.025: #condición sobre 1ª diferencia
            pts[i]=pts[i]+1
        prop_neg=sum(diff2[:i]<0)/i #proporción de 2as dif negativas hasta k
        if prop_neg > 0.8:
            pts[i]=pts[i]+1
        #prop de cubrimientos marginales anteriores mayores que el actual
        prop_exp_mayor=sum(prop_exp[:i]>=prop_exp[i-1])/i
        if prop_exp_mayor > 0.7:
            pts[i]=pts[i]+1
        #condicion sobre valores restantes de la 2a dif
        if i < diff2.shape[0]-1 and min(diff2[i:]) > -0.01:
            pts[i]=pts[i]+1
        ratio_exp=prop_exp[i]/prop_exp[i-1] #ratio de cubrimientos marginales
        if ratio_exp > 2.3:
            pts[i]=pts[i]+1
        if i < diff2.shape[0]-1:
            ratio_diff2=diff2[i-1]/diff2[i] #ratio de 2as dif
            if abs(ratio_diff2) > 5:
                pts[i]=pts[i]+1
        #ratio de 1a dif actual respecto a max de dif restantes
        if diff[i-1]/max(diff[i:]) > 2: 
            pts[i]=pts[i]+1
        #ratio de 2a dif actual respecto a min de 2as dif restantes
        if i < diff2.shape[0]-1 and diff2[i-1]/min(diff2[i:]) > 3:
            pts[i]=pts[i]+1
    return np.argmax(pts)+nmin,pts
        
import math
def conds_score2(gci_,id,u,p=None,c=None,b=None):
    
    if "nan"==str(id):
        return np.NAN

    k=gci_.shape[0]
    s_c=1-gci_ #proporción sin cubrimiento total
    d=np.diff(gci_)
    d2=np.diff(d)
    p_e=d/s_c[:-1] #proporciones que se cubren en cada k
   
    pts=np.zeros(k)

    if p is None and c is None:
        pts[0]=9-(b[0]*1+b[1]*2)
        for i in range(1,d.shape[0]):
            pts[i]=i/d.shape[0] #fracción creciente
            
            r_d=d[i-1]/d[i] #condición sobre ratio de diferencias
            r_l=(gci_[i]-gci_[i-1])/gci_[i-1] #ratio relativo
            p_n=sum(d2[:i]<0)/i #proporción de 2as dif negativas hasta k
            p_e_m=sum(p_e[:i]>=p_e[i-1])/i #prop de cubrimientos marginales anteriores mayores que el actual
            r_e=p_e[i-1]/p_e[i] #ratio de cubrimientos marginales

            if i < d2.shape[0]-1: 
                r_d2=d2[i-1]/d2[i] #ratio de 2as dif

            if r_d > u[0]: 
                pts[i]+=1
                
            if r_l > u[1]: 
                pts[i]+=1

            if d[i-1]> u[2]: 
                pts[i]+=1

            if p_n > u[3]: 
                pts[i]+=1
            
            if p_e_m > u[4]: 
                pts[i]+=1

            #condicion sobre valores restantes de la 2a dif
            if i < d2.shape[0]-1 and min(d2[i:]) > u[5]: 
                pts[i]+=1
            
            if r_e > u[6]: pts[i]+=1
            
            if i < d2.shape[0]-1 and abs(r_d2) > u[7]: 
                pts[i]+=1
                
            #ratio de 1a dif actual respecto a max de dif restantes
            if d[i-1]/max(d[i:]) > u[8]: 
                pts[i]+=1
                
            #ratio de 2a dif actual respecto a min de 2as dif restantes
            if i < d2.shape[0]-1 and d2[i-1]/min(d2[i:]) >u[9]:
                pts[i]+=1
                
        return np.argmax(pts)+1
    
    elif p is not None:
        pts[0]=1
        for i in range(1,d.shape[0]):
                        
            r_d=d[i-1]/d[i] #condición sobre ratio de diferencias
            r_l=(gci_[i]-gci_[i-1])/gci_[i-1] #ratio relativo
            p_n=sum(d2[:i]<0)/i #proporción de 2as dif negativas hasta k
            p_e_m=sum(p_e[:i]>=p_e[i-1])/i #prop de cubrimientos marginales anteriores mayores que el actual
            r_e=p_e[i-1]/p_e[i] #ratio de cubrimientos marginales
            if i < d2.shape[0]-1: 
                r_d2=d2[i-1]/d2[i] #ratio de 2as dif

            if r_d > u[0]: 
                pts[i]+=p[0]
                
            if r_l > u[1]: 
                pts[i]+=p[1]

            if d[i-1]> u[2]: 
                pts[i]+=p[2]

            if p_n > u[3]: 
                pts[i]+=p[3]
            
            if p_e_m > u[4]: 
                pts[i]+=p[4]

            #condicion sobre valores restantes de la 2a dif
            if i < d2.shape[0]-1 and min(d2[i:]) > u[5]: 
                pts[i]+=p[5]
            
            if r_e > u[6]: pts[i]+=p[6]
            
            if i < d2.shape[0]-1 and abs(r_d2) > u[7]: 
                pts[i]+=p[7]
                
            #ratio de 1a dif actual respecto a max de dif restantes
            if d[i-1]/max(d[i:]) > u[8]: 
                pts[i]+=p[8]
                
            #ratio de 2a dif actual respecto a min de 2as dif restantes
            if i < d2.shape[0]-1 and d2[i-1]/min(d2[i:]) >u[9]:
                pts[i]+=p[9]
                
            if np.abs(pts[i]-1.0) < 1e-15: pts[i]=1.0
            elif np.abs(pts[i]-2.0) < 1e-15: pts[i]=2.0
            elif np.abs(pts[i]-3.0) < 1e-15: pts[i]=3.0
            elif np.abs(pts[i]-4.0) < 1e-15: pts[i]=4.0
            elif np.abs(pts[i]-5.0) < 1e-15: pts[i]=5.0
            elif np.abs(pts[i]-6.0) < 1e-15: pts[i]=6.0
            elif np.abs(pts[i]-7.0) < 1e-15: pts[i]=7.0
            elif np.abs(pts[i]-8.0) < 1e-15: pts[i]=8.0
            elif np.abs(pts[i]-9.0) < 1e-15: pts[i]=9.0
            elif np.abs(pts[i]-10.0) < 1e-15: pts[i]=10.0
                
        return np.amax(np.asarray(pts==np.amax(pts)).nonzero())+1
    
    elif c is not None:
        pts[0]=np.amax([np.sum(c)-(b[0]*1+b[1]*2),1])
        for i in range(1,d.shape[0]):
            pts[i]=i/d.shape[0] #fracción creciente
            
            r_d=d[i-1]/d[i] #condición sobre ratio de diferencias
            r_l=(gci_[i]-gci_[i-1])/gci_[i-1] #ratio relativo
            p_n=sum(d2[:i]<0)/i #proporción de 2as dif negativas hasta k
            p_e_m=sum(p_e[:i]>=p_e[i-1])/i #prop de cubrimientos marginales anteriores mayores que el actual
            r_e=p_e[i-1]/p_e[i] #ratio de cubrimientos marginales

            if i < d2.shape[0]-1: 
                r_d2=d2[i-1]/d2[i] #ratio de 2as dif

            if r_d > u[0] and c[0]==1:
                pts[i]+=1
                
            if r_l > u[1] and c[1]==1:
                pts[i]+=1

            if d[i-1]> u[2] and c[2]==1:
                pts[i]+=1

            if p_n > u[3] and c[3]==1: 
                pts[i]+=1
            
            if p_e_m > u[4] and c[4]==1: 
                pts[i]+=1

            #condicion sobre valores restantes de la 2a dif
            if i < d2.shape[0]-1 and min(d2[i:]) > u[5] and c[5]==1: 
                pts[i]+=1
            
            if r_e > u[6] and c[6]==1: pts[i]+=1
            
            if i < d2.shape[0]-1 and abs(r_d2) > u[7] and c[7]==1: 
                pts[i]+=1
                
            #ratio de 1a dif actual respecto a max de dif restantes
            if d[i-1]/max(d[i:]) > u[8] and c[8]==1: 
                pts[i]+=1
                
            #ratio de 2a dif actual respecto a min de 2as dif restantes
            if i < d2.shape[0]-1 and d2[i-1]/min(d2[i:]) >u[9] and c[9]==1:
                pts[i]+=1
        
        return np.argmax(pts)+1

    


def conds_score(gci_,id,u,p=None,c=None):
    
    if "nan"==str(id):
        return np.NAN

    k=gci_.shape[0]
    s_c=1-gci_ #proporción sin cubrimiento total
    d=np.diff(gci_)
    d2=np.diff(d)
    p_e=d/s_c[:-1] #proporciones que se cubren en cada k
   
    pts=np.zeros(k)

    if p is None and c is None:
        pts[0]=7
        for i in range(1,d.shape[0]):
            pts[i]=i/d.shape[0] #fracción creciente
            
            r_d=d[i-1]/d[i] #condición sobre ratio de diferencias
            r_l=(gci_[i]-gci_[i-1])/gci_[i-1] #ratio relativo
            p_n=sum(d2[:i]<0)/i #proporción de 2as dif negativas hasta k
            p_e_m=sum(p_e[:i]>=p_e[i-1])/i #prop de cubrimientos marginales anteriores mayores que el actual
            r_e=p_e[i-1]/p_e[i] #ratio de cubrimientos marginales

            if i < d2.shape[0]-1: 
                r_d2=d2[i-1]/d2[i] #ratio de 2as dif

            if r_d > u[0]: 
                pts[i]+=1
                
            if r_l > u[1]: 
                pts[i]+=1

            if d[i-1]> u[2]: 
                pts[i]+=1

            if p_n > u[3]: 
                pts[i]+=1
            
            if p_e_m > u[4]: 
                pts[i]+=1

            #condicion sobre valores restantes de la 2a dif
            if i < d2.shape[0]-1 and min(d2[i:]) > u[5]: 
                pts[i]+=1
            
            if r_e > u[6]: pts[6]+=1
            
            if i < d2.shape[0]-1 and abs(r_d2) > u[7]: 
                pts[i]+=1
                
            #ratio de 1a dif actual respecto a max de dif restantes
            if d[i-1]/max(d[i:]) > u[8]: 
                pts[i]+=1
                
            #ratio de 2a dif actual respecto a min de 2as dif restantes
            if i < d2.shape[0]-1 and d2[i-1]/min(d2[i:]) >u[9]:
                pts[i]+=1
    elif p is not None:
        pts[0]=p[0]
        for i in range(1,d.shape[0]):
            pts[i]=i/d.shape[0] #fracción creciente
            
            r_d=d[i-1]/d[i] #condición sobre ratio de diferencias
            r_l=(gci_[i]-gci_[i-1])/gci_[i-1] #ratio relativo
            p_n=sum(d2[:i]<0)/i #proporción de 2as dif negativas hasta k
            p_e_m=sum(p_e[:i]>=p_e[i-1])/i #prop de cubrimientos marginales anteriores mayores que el actual
            r_e=p_e[i-1]/p_e[i] #ratio de cubrimientos marginales

            if i < d2.shape[0]-1: 
                r_d2=d2[i-1]/d2[i] #ratio de 2as dif

            if r_d > u[0]: 
                pts[i]+=p[1]
                
            if r_l > u[1]: 
                pts[i]+=p[2]

            if d[i-1]> u[2]: 
                pts[i]+=p[3]

            if p_n > u[3]: 
                pts[i]+=p[4]
            
            if p_e_m > u[4]: 
                pts[i]+=p[5]

            #condicion sobre valores restantes de la 2a dif
            if i < d2.shape[0]-1 and min(d2[i:]) > u[5]: 
                pts[i]+=p[6]
            
            if r_e > u[6]: pts[6]+=p[7]
            
            if i < d2.shape[0]-1 and abs(r_d2) > u[7]: 
                pts[i]+=p[8]
                
            #ratio de 1a dif actual respecto a max de dif restantes
            if d[i-1]/max(d[i:]) > u[8]: 
                pts[i]+=p[9]
                
            #ratio de 2a dif actual respecto a min de 2as dif restantes
            if i < d2.shape[0]-1 and d2[i-1]/min(d2[i:]) >u[9]:
                pts[i]+=p[10]
    elif c is not None:
        pts[0]=np.sum(c)*0.7
        for i in range(1,d.shape[0]):
            pts[i]=i/d.shape[0] #fracción creciente
            
            r_d=d[i-1]/d[i] #condición sobre ratio de diferencias
            r_l=(gci_[i]-gci_[i-1])/gci_[i-1] #ratio relativo
            p_n=sum(d2[:i]<0)/i #proporción de 2as dif negativas hasta k
            p_e_m=sum(p_e[:i]>=p_e[i-1])/i #prop de cubrimientos marginales anteriores mayores que el actual
            r_e=p_e[i-1]/p_e[i] #ratio de cubrimientos marginales

            if i < d2.shape[0]-1: 
                r_d2=d2[i-1]/d2[i] #ratio de 2as dif

            if r_d > u[0] and c[0]==1:
                pts[i]+=1
                
            if r_l > u[1] and c[1]==1:
                pts[i]+=1

            if d[i-1]> u[2] and c[2]==1:
                pts[i]+=1

            if p_n > u[3] and c[3]==1: 
                pts[i]+=1
            
            if p_e_m > u[4] and c[4]==1: 
                pts[i]+=1

            #condicion sobre valores restantes de la 2a dif
            if i < d2.shape[0]-1 and min(d2[i:]) > u[5] and c[5]==1: 
                pts[i]+=1
            
            if r_e > u[6] and c[6]==1: pts[6]+=1
            
            if i < d2.shape[0]-1 and abs(r_d2) > u[7] and c[7]==1: 
                pts[i]+=1
                
            #ratio de 1a dif actual respecto a max de dif restantes
            if d[i-1]/max(d[i:]) > u[8] and c[8]==1: 
                pts[i]+=1
                
            #ratio de 2a dif actual respecto a min de 2as dif restantes
            if i < d2.shape[0]-1 and d2[i-1]/min(d2[i:]) >u[9] and c[9]==1:
                pts[i]+=1

    return np.argmax(pts)+1

