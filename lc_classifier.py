import sys
sys.path.append(r'/home/christian/Documents/Bachelor/numerics')
from physical_functions_mf import *
from add_solutions_functions import *
from scipy.optimize import root
from multiprocessing import Pool
from time import time, sleep
from random import random
t1 = time()
root_path = '/home/christian/Documents/Bachelor/numerics/mean_field/4.4k13_1.5w5.5_3d/'
name = '_4.4k13_1.5w5.5'
namenp = name + '.npy'



# for steps in range(7):
#     try:
#         trajs_mspace = np.load(root_path+'mphase_traj_271.npz')['arr_0']
#         break
#     except:
#         sleep(40*60)

print('loaded')
def traj_dist(diff_max,a,b):
    endcut = int(4000)
    res = np.ones(diff_max)
    res[0]= np.sum(np.log(np.abs(a[-endcut:]-b[-endcut:])))/endcut
    for i in range(1,diff_max):
        res[i] = np.sum(np.log(np.abs(a[-endcut+i:]-b[-endcut:-i])))/(endcut-i)
    return res


def classify_traj(trajs):
    max_offset = 2000
    middle= trajs.shape[0]//2
    traj_classes = [trajs[middle,middle]]
    phase_map = np.zeros((trajs.shape[0],trajs.shape[1]))
    for i in range(trajs.shape[0]):
        for j in range(trajs.shape[1]):
            if i==middle and j==middle:
                continue
            elif np.isnan(trajs[i,j,0,1]):
                phase_map[i,j]=-1
                continue
            found = 0
            for c_index in range(len(traj_classes)):
                distance = traj_dist(max_offset,trajs[i,j,:,1],traj_classes[c_index][:,1])
                mind = min(distance)
                if mind<-4:
                    phase_map[i,j] = c_index
                    found = 1
                    break
            if found==0:
                phase_map[i,j] = c_index+1
                traj_classes.append(trajs[i,j])
                
    return traj_classes, phase_map

# classes , phase_map = classify_traj(trajs_mspace)

# np.savez_compressed(root_path+'lc_classes_271',classes)
# np.savez_compressed(root_path+'lc_mphasemap_271',phase_map)
# duration = time()-t1
# print(f"{duration//3600}h, {(duration%3600)//60}min, {duration%60}s")