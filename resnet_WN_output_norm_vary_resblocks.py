# ResNet, fixed resblock width, varying number of resblocks
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import pickle as pkl
import math

'''
Experiments for ResNet network output norm, fixed resblock width, varying number of resblocks (figure 7)
'''

init = ['he', 'glorot', 'gaussian1'][0]

N=2000 # number of samples
n=300 # dim of each sample
m = 100 # width of network
B_list = [3,5,10,20,100] # number of resblocks
print('width: ',m)


for scaled in [ False,True]: # True is using proposed WN initialization, False for traditional WN
    file_name="results/resnet_varyB_init-{}_N{}_n{}_m{}_WN_scaled-{}.pkl".format(init,N,n,m,int(scaled))
    print(file_name)

    epsilon_B_fwd = [ ]
    for B in B_list:
        epsilon_N = []
        W = {} # dictionary of weight matrix
        for l in range(B+1):
            n1 = n if l==0 else m
            if init=='he':
                W[l] = [np.random.randn(m,n)/np.sqrt(0.5*m) if l==0 else np.random.randn(m,m)/np.sqrt(0.5*m),
                        np.random.randn(m,m)/np.sqrt(0.5*m)]
            elif init=='glorot':
                W[l] = [np.random.randn(m,n)/np.sqrt(m) if l==0 else np.random.randn(m,m)/np.sqrt(m),
                        np.random.randn(m,m)/np.sqrt(m)]
            elif init=='gaussian1':
                W[l] = [np.random.randn(m,n) if l==0 else np.random.randn(m,m),
                        np.random.randn(m,m)]
            norm = np.linalg.norm(W[l][0], axis=1)
            W[l][0] /= norm[:,None]
            norm = np.linalg.norm(W[l][1], axis=1)
            W[l][1] /= norm[:,None]

        for N_i in tqdm.tqdm(range(N)):
            x = np.random.randn(n)
            h=1.*x
            # fwd pass
            R=1.*W[0][0]
            if scaled:
                R *= np.sqrt(2*R.shape[1]/R.shape[0])
            a = R.dot(h)  
            h = np.maximum(0,a)
            h_res = 1.*h
            for l in range(B):
                R=1.*W[l+1][0]
                if scaled:
                    R *= np.sqrt(2*R.shape[1]/R.shape[0])
                a = R.dot(h_res)  
                h_res = np.maximum(0,a)
                R=1.*W[l+1][1]
                if scaled:
                    R *= np.sqrt(2*R.shape[1]/R.shape[0])
                a = R.dot(h_res)  
                h_res = np.maximum(0,a)
            if scaled:
                h_res /= B
            h = h + h_res
            epsilon_N.append( np.linalg.norm(h)/np.linalg.norm(x) )
        mn,std = np.mean(epsilon_N), np.std(epsilon_N)
        epsilon_B_fwd.append((mn,std))
        print('B {}, mn {}, std {}'.format(B,mn,std))

    with open(file_name, "wb") as f:
        pkl.dump((B_list, epsilon_B_fwd), f)
