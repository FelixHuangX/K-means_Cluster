# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 18:38:20 2019

@author: 17683746951
"""
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import pandas as pd

def bestcenters(data,k,C):
    m = data.shape[0]
    D = []
    for i in range(m):
        Dis = []
        for j in range(k):
            dis = math.sqrt((data[i][0]-C[j][0])**2+(data[i][1]-C[j][1])**2)
            Dis.append(dis)
        n = Dis.index(min(Dis))
        D.append([n,data[i][0],data[i][1]])
    
    DS = np.array(sorted(D,key=lambda x:x[0]))
    
    #现在已经有了每一个样本属于某个类的二维数组DS，现在要对每一类的样本求取中心坐标
    CC = []
    for i in range(k):
        tindex = np.where(DS[:,0]==i)
        tep = DS[tindex]
        CC.append([np.mean(tep[:,1]),np.mean(tep[:,2])])
    
    return DS,CC
    


def bestcluster(data,R,e,k):
    
    DS,C = bestcenters(data,k,R)
    C = np.array(C)
    m = C.shape[0]
    distance = math.sqrt((R[0][0]-C[0][0])**2+(R[0][1]-C[0][1])**2)
    E = distance*k    
    
    while E > e:
        R = C
        DS,C = bestcenters(data,k,R)
        distance = 0
        for i in range(m):
            distance += math.sqrt((R[i][0]-C[i][0])**2+(R[i][1]-C[i][1])**2)
        E = distance

    return DS,C,E

if __name__ == "__main__":
    rawdata = open('rawdata.txt')
    data = []
    while True:
        lines = rawdata.readline()
        if not lines:
            break
        l,r = [float(i) for i in lines.split()]
        data.append([l,r])
    
    data = np.array(data)
    m,n = data.shape
    #初始化类中心
    #根据密度据类处理的结果，分5个类效果比较好
    k = 5
    R = [] 
    for i in range(k):
        r = random.randint(0,m-1)
        axis = data[r][:]
        R.append(axis)
    
    #设置迭代误差
    e = 0.001    
    DS,C,E = bestcluster(data,R,e,k)
    DS = pd.DataFrame(DS)
    print(C)
    print(E)
    
    
    S = np.array(DS.drop([0], axis=1))
    S_target = np.array(DS[0])
    target = []
    for i in range(len(S_target)):
        target.append(int(S_target[i]))
    target = pd.DataFrame(target)
    
    plt.figure()
    plt.scatter(data[:,0],data[:,1])
    plt.title('raw dataset')
    plt.show()
    
    
    plt.figure()
    #'navy':深蓝色，‘turquoise’:绿松石，‘darkorange’:暗桔色，'azure':天蓝色，'brown'：棕色，‘burlylood’:实木色,'red'：红色，‘yellow’:黄色,'purple':紫色
    #'pink'：粉色，'green':绿色
    colors = ['navy', 'turquoise', 'darkorange', 'black', 'brown']
    lw = 2
    
    #cluster_name = ['o07','o21','norm','b07','b14']
    
    for color, i in zip(colors, [0, 1, 2, 3, 4]):
        plt.scatter(S[np.where(target==i)[0],0], S[np.where(target==i)[0],1], color=color, alpha=.8, lw=lw)
        
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('Cluster result of raw dataset')
    plt.show()
