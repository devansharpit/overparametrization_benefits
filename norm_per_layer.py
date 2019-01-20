import numpy as np
import tqdm
import matplotlib.pyplot as plt
import pickle as pkl
import math

'''
Experiments for Fig 1 and 2-- activation norm and gradient norm for un-normalized deep ReLU networks

'''

def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum()

# Un-normalized ReLU net
init = ['he', 'glorot'][0]



N=2000 # number of samples
n=500 # dim of each sample
L=10 # depth of network
m =  100 # width of network
print('width: ',m)
target_dim = 20

file_name="results/norm_per_layer_init_{}_L{}_N{}_n{}_m{}.pkl".format(init,L,N,n,m)
print(file_name)

W = {} # dictionary of weight matrix
for l in range(L):
	n1 = n if l==0 else m
	if init=='he':
	    W[l] = np.random.randn(m,n)/np.sqrt(0.5*m) if l==0 else np.random.randn(m,m)/np.sqrt(0.5*m)
	elif init=='glorot':
	    W[l] = np.random.randn(m,n)/np.sqrt(m) if l==0 else np.random.randn(m,m)/np.sqrt(m)
W_softmax = np.random.randn(target_dim,m)

epsilon_N_fwd = [ [] for _ in range(L) ]
epsilon_L_fwd = []

epsilon_N_bwd = [ [] for _ in range(L) ]
epsilon_L_bwd = []


epsilon_N = []
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
        epsilon_N_fwd[l].append( np.linalg.norm(h)/np.linalg.norm(x) )

    logit = W_softmax.dot(h)
    pred = softmax(logit)
    grad_err = pred - target # derivative for log loss
    delta_x_y = np.asarray(H[-1]>0., dtype='float')* W_softmax.T.dot(grad_err)
    grad_a = 1.* delta_x_y
    # bwd pass
    for l in range(L):
        grad_W_norm = np.linalg.norm(H[-l-2])* np.linalg.norm(grad_a)
        epsilon_N_bwd[-l-1].append( grad_W_norm/(np.linalg.norm(x)* np.linalg.norm(delta_x_y)) )
        if l<L:
            R=1.*W[L-l-1].T
            grad_a = np.asarray(H[-l-2]>0., dtype='float')* R.dot(grad_a)
    	

for l in range(L):
	mn,std = np.mean(epsilon_N_fwd[l]), np.std(epsilon_N_fwd[l])
	epsilon_L_fwd.append((mn,std))
	mn,std = np.mean(epsilon_N_bwd[l]), np.std(epsilon_N_bwd[l])
	epsilon_L_bwd.append((mn,std))

print(epsilon_L_fwd)	
print(epsilon_L_bwd)	
with open("results/norm_per_layer_init_{}_L{}_N{}_n{}_m{}.pkl".format(init,L,N,n,m), "wb") as f:
	pkl.dump((epsilon_L_fwd, epsilon_L_bwd), f)

