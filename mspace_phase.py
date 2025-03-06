import sys
sys.path.append(r'/home/christian/Documents/Bachelor/numerics')
from physical_functions_mf import *
from add_solutions_functions import *
from scipy.optimize import root
from multiprocessing import Pool
from time import time
from random import random
t1 = time()
root_path = '/home/christian/Documents/Bachelor/numerics/mean_field/4.4k13_1.5w5.5_3d/'
name = '_4.4k13_1.5w5.5'
namenp = name + '.npy'


k = np.load(root_path+ 'kparam' + namenp)
w = np.load(root_path+ 'wparam' + namenp)
d = np.load(root_path+ 'dparam' + namenp)
def find_index(kv,wv,dv):
    kind =0
    wind = 0
    dind =0
    for ik, kval in enumerate(k):
        if kval>=kv:
            kind=ik
            break
    for iw, wval in enumerate(w):
        if wval>=wv:
            wind=iw
            break
    for id, dval in enumerate(d):
        if dval>=dv:
            dind = id
            break
    return kind, wind, dind


ik, iw, id = find_index(10.993, 2.388, -2.14)
kval, wval, dval = k[ik], w[iw], d[id]
print(kval,wval,dval)

alpha1 = np.pi/6
alpha2 = np.pi/3

def rma_x(alpha):
    row1 = [1,0,0]
    row2 = [0,np.cos(alpha),-np.sin(alpha)]
    row3 = [0,np.sin(alpha),np.cos(alpha)]
    return np.array([row1,row2,row3])

def rma_y(alpha):
    row1 = [np.cos(alpha),0,-np.sin(alpha)]
    row2 = [0,1,0]
    row3 = [np.sin(alpha),0,np.cos(alpha)]
    return np.array([row1,row2,row3])

def rma_z(alpha):
    row1 = [np.cos(alpha),-np.sin(alpha),0]
    row2 = [np.sin(alpha),np.cos(alpha),0]
    row3 = [0,0,1]
    return np.array([row1,row2,row3])

gridpoints = 271
cut1 = np.zeros((gridpoints,gridpoints,3))
middle = cut1.shape[0]//2
for i in range(cut1.shape[0]):
    for j in range(cut1.shape[1]):
        cut1[i,j,0]=i
        cut1[i,j,1]=j
        
cut1[:,:,0]-=middle
cut1[:,:,1]-=middle
cut1/=middle/0.5
roty = rma_y(alpha1)
rotx = rma_x(alpha2)
cut1 = np.tensordot(cut1,roty,[2,1])
cut1 = np.tensordot(cut1,rotx,axes=[2,1])

numb_of_tsteps = 10000
offset = 16000

def average(traj):  
    pfound = 0
    counter =0
    start = 0
    while pfound ==0:
        for j in range(traj.shape[1]-1,start,-1):
        # print(np.linalg.norm(traj[:,0]-traj[:,j]))
            if np.linalg.norm(traj[:,start]-traj[:,j])<1e-3:
                break
        if j ==start+1:
            # print('next_try')
            counter +=1
            if counter==15:
                # print('abortion')
                return np.mean(traj[:,start:],axis=1), pfound
            start +=100
            
        else:
            pfound =1
            break
    return np.mean(traj[:,start:j],axis=1)#, pfound

def integrate(index):
    i, j = index//gridpoints, index%gridpoints
    arguments = wval,kval, dval, 1, 0.2 
    if np.linalg.norm(cut1[i,j])<=0.5:
        traj = solve_ivp(gl2,(0,20000),y0=cut1[i,j],args=arguments,method='DOP853',t_eval=np.linspace(offset,20000,numb_of_tsteps),rtol=1e-10)#,atol=1e-25)
        return np.mean(traj.y,axis=1), average(traj.y)
        # return np.row_stack((traj.t,traj.y))
    else:
        return np.ones(3),np.ones(3)
        # return np.ones((4,numb_of_tsteps))*np.nan
po = Pool(9)
means = np.zeros((gridpoints,gridpoints,3))
avers = np.zeros_like(means)
# trajs = np.zeros((gridpoints,gridpoints,numb_of_tsteps,4))
result = po.imap(integrate,range(gridpoints**2),chunksize=100)
for index, res in enumerate(result):
    i, j = index//gridpoints, index%gridpoints
    means[i,j], avers[i,j] = res
    # trajs[i,j] = res.transpose()

np.save(root_path+'DOP_mphase_mean_tilt_271',means)
np.save(root_path+'DOP_mphase_aver_tilt_271',avers)
duration = time()-t1
print(f"{duration//3600}h, {(duration%3600)//60}min, {duration%60}s")