import sys
sys.path.append(r'/home/christian/Documents/Bachelor/numerics')
from physical_functions_mf import *
from add_solutions_functions import *
from scipy.optimize import root
from multiprocessing import Pool
from time import time, sleep
from random import random
import gc
import os
t1 = time()

# while True:
    
#     if os.path.exists('/home/christian/Documents/Bachelor/numerics/mean_field/4.4k13_1.5w5.5_3d/means_750line_10k_1.85d1.66.npy'):
#         sleep(2*60)
#         break
#     sleep(60*60)
print('start')
root_path = '/home/christian/Documents/Bachelor/numerics/mean_field/4.4k13_1.5w5.5_3d/'
name = '_750line_10k_1.76d1.73'
namenp = name + '.npy'

gridpoints = 750

k = 10.118909853249477
# k = 10.60930817610063 
# k = 11.896603773584905
# d= np.linspace(-1.85, -1.66,gridpoints)
# d= np.linspace(-2.2,-1.63,gridpoints)
# d= np.linspace(-2.3,-2.11,gridpoints)
# d= np.linspace(-1.725,-1.7235,gridpoints)
# d = np.linspace(-1.7242,-1.72403,gridpoints)
# d = np.linspace(-2.694, -2.64,gridpoints)
d = np.linspace(-1.76,-1.73,gridpoints)
w = 2.3889


def fill_binary(length,numb):
    diff = length -len('{:b}'.format(numb))
    if diff:
        return diff*'0'
    else:
        return ''

def starting_grid(expansion,resolution):
    # points = np.zeros((resolution,resolution,resolution,3))
    flat = []
    stepsize = expansion/(resolution-1)
    for i in range(resolution):
        for j in range(resolution):
            for l in range(resolution):
                a = 0b000
                for n in range(7):
                    b = '{}{:b}'.format(fill_binary(3,a),a)
                    b= np.array([*b],dtype=np.int32)
                    a +=1
                    # print(1-2*b)
                    # print(np.array([i,j,l])*(1-2*b))
                    p1 = np.array([i,j,l])*(1-2*b)*stepsize
                    if np.linalg.norm(p1)<=0.5:
                        flat.append(p1)
    out = [flat[0]]
    for i in range(1,len(flat)):
        take = 1
        for j in range(len(out)):
            if np.linalg.norm(flat[i]-out[j])==0:
                # print('sorted out')
                take =0
        if take:
            out.append(flat[i])

    return np.row_stack(out)


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
    return np.mean(traj[:,start:j],axis=1), pfound

start_points = starting_grid(0.62,4)
numb_of_start = int(start_points.shape[0])
numb_of_tsteps = int(10000)

def stab_find(kval,wval,dval,sgam):
    arguments = wval,kval, dval, Gam, sgam 
    trajs = []
    means = np.zeros((3,numb_of_start))
    averages = np.zeros((3,numb_of_start))
    found_p = np.zeros(numb_of_start)
    for i,s in enumerate(start_points):
        # traj = solve_ivp(gl2,(0,1400),y0=s,args=arguments,method='LSODA',t_eval=np.linspace(1000,1399,numb_of_tsteps))#np.arange(numb_of_tsteps)/numb_of_tsteps*10000)#,rtol=1e-12,atol=1e-25)
        traj = solve_ivp(gl2,(0,20000),y0=s,args=arguments,method='DOP853',t_eval=np.linspace(16000,19999,numb_of_tsteps),rtol=1e-10)
        # ind = int(3/4*len(traj.t))
        trajs.append(traj)
        averages[:,i], found_p[i]=average(traj.y)
        means[:,i] = np.mean(traj.y,axis=1) 
    return means, averages, trajs, found_p




def endp_wrapper(index):
    index = int(index)
    
    
    trajectoriesw=np.ones((numb_of_start,numb_of_tsteps,4))
    
    if True:# stability_plus[i,w_cut,j]==0 and k[i]>9.4 and d[j]<0:
        solw, aver ,trajectw, p_found= stab_find(k,w,d[index],0.2)
        for it in range(len(trajectw)):
            trajectoriesw[it,:,0] = trajectw[it].t
            trajectoriesw[it,:,1:] = trajectw[it].y.transpose()
    
    return solw, aver #, trajectoriesw

subgrid = gridpoints//3

w_cut_aver = np.zeros((gridpoints,3,numb_of_start))
w_cut_means = np.zeros_like(w_cut_aver)
# w_cut_traj = np.zeros((gridpoints,numb_of_start,numb_of_tsteps,4))
po = Pool(9)
result = po.imap(endp_wrapper,np.arange(gridpoints),chunksize=10)
for index, res in enumerate(result):
    # w_cut_means[index],w_cut_aver[index],w_cut_traj[index] = res
    w_cut_means[index],w_cut_aver[index]= res

# np.savez_compressed(root_path+f'w_cut_traj_200line',w_cut_traj)
np.save(root_path+'aver'+name,w_cut_aver)
np.save(root_path+'means'+name,w_cut_means)

duration = time()-t1
print("{:.0f}h, {:.0f}min, {:.1f}s".format(duration//3600, (duration%3600)//60, duration%60))


print('finished')