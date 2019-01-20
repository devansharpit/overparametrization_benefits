import numpy as np
import tqdm
import matplotlib.pyplot as plt
import pickle as pkl
import math

'''
Experiments for Fig 6

'''

def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum()

# Un-normalized ReLU net
init = ['he', 'glorot'][0]



N=2000 # number of samples
n=500 # dim of each sample
L=10 # depth of network
delta = 0.05 # failure probability
# epsilon = 0.15
# epsilon_prime = (1. + epsilon)**(1./L) - 1.
# m =  100 # int((4./epsilon_prime)* math.log(4.*N*L/delta)) # width of network
# print('width: ',m)
target_dim = 20

file_name="results/norm_per_layer_init_{}_L{}_N{}_n{}.pkl".format(init,L,N,n)
print(file_name)




m_list = [500, 1000, 2000, 4000]
epsilon_N_fwd = {}
epsilon_N_bwd = {}
for m in m_list:
    epsilon_N_fwd[m] = []
    epsilon_N_bwd[m] = []


for m in m_list:
    print('On ',m)

    W = {} # dictionary of weight matrix
    for l in range(L):
        n1 = n if l==0 else m
        if init=='he':
            W[l] = np.random.randn(m,n)/np.sqrt(0.5*m) if l==0 else np.random.randn(m,m)/np.sqrt(0.5*m)
        elif init=='glorot':
            W[l] = np.random.randn(m,n)/np.sqrt(m) if l==0 else np.random.randn(m,m)/np.sqrt(m)
    W_softmax = np.random.randn(target_dim,m)

    for N_i in tqdm.tqdm(range(N)):
        x = np.random.randn(n)
        target = np.random.rand(target_dim)
        target /= target.sum()
        H = []
        H.append(x)
        h=1.*x
        # fwd pass
        for l in range(L):
            R=1.*W[l]
            a = R.dot(h)  
            h = np.maximum(0,a)
            H.append(h)
            epsilon_N_fwd[m].append( np.abs( 1. - np.linalg.norm(h)/np.linalg.norm(x) ) )

        logit = W_softmax.dot(h)
        pred = softmax(logit)
        grad_err = pred - target # derivative for log loss
        delta_x_y = np.asarray(H[-1]>0., dtype='float')* W_softmax.T.dot(grad_err)
        grad_a = 1.* delta_x_y
        # bwd pass
        for l in range(L):
            grad_W_norm = np.linalg.norm(H[-l-2])* np.linalg.norm(grad_a)
            epsilon_N_bwd[m].append( np.abs( 1. - grad_W_norm/(np.linalg.norm(x)* np.linalg.norm(delta_x_y)) ) )
            if l<L:
                R=1.*W[L-l-1].T
                grad_a = np.asarray(H[-l-2]>0., dtype='float')* R.dot(grad_a)
    	
epsilon_m_fwd = []
epsilon_m_bwd = []
for m in m_list:
	mn,std = np.mean(epsilon_N_fwd[m]), np.std(epsilon_N_fwd[m])
	epsilon_m_fwd.append((mn,std))
	mn,std = np.mean(epsilon_N_bwd[m]), np.std(epsilon_N_bwd[m])
	epsilon_m_bwd.append((mn,std))

print(epsilon_m_fwd)	
print(epsilon_m_bwd)	
with open("results/bound_comparison_empirical_init_{}_L{}_N{}_n{}.pkl".format(init,L,N,n), "wb") as f:
	pkl.dump((epsilon_m_fwd, epsilon_m_bwd), f)

m=np.arange(500, 4000)
epsilon = lambda x: (1. + (4./x)*np.log(4.*N*L/delta) )**L -1.
ep_list = [epsilon(m_i) for m_i in m]
epsilon_m_fwd = ep_list
epsilon_m_bwd = ep_list
with open("results/bound_comparison_theoretical_init_{}_L{}_N{}_n{}.pkl".format(init,L,N,n), "wb") as f:
    pkl.dump((epsilon_m_fwd, epsilon_m_bwd), f)


