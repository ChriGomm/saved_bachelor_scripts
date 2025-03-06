import sys
sys.path.append(r'/home/christian/Documents/Bachelor/numerics')
from physical_functions_mf import *
from add_solutions_functions import *
from scipy.optimize import root
from multiprocessing import Pool
from time import time
t1 = time()
root_path = '/home/christian/Documents/Bachelor/numerics/mean_field/4.4k13_1.5w5.5_3d/'
name = '_4.4k13_1.5w5.5'
namenp = name + '.npy'

solnumb_file = root_path+ f'solnumb' + namenp
xsol_file = root_path+f'xsol' +namenp
ysol_file = root_path+f'ysol' +namenp

# params = np.load(root_path+'params71.npy')
# print('worked')
k = np.load(root_path+ 'k_cut_traj4.npz')
w = np.load(root_path+ 'w_cut_traj4.npz')
d = np.load(root_path+ 'd_cut_traj4.npz')



stability_plus = np.load(root_path+'stability_plus.npy')
stability = np.load(root_path+'stability.npy')

def average(traj):
    start_index = int(3/4*traj.shape[0])
    start= traj[start_index,:]
    for j in range(traj.shape[0]-1,start_index,-1):
        if np.linalg.norm(start-traj[j,:])<1e-5:
            break
    return np.mean(traj[start_index:j,:],axis=0)

def av_wrap(ind,k,gridy):
    i, j = ind//gridy, ind%gridy
    trajs = k[i,j]
    mean = np.zeros((3,trajs.shape[0]))
    for n in range(trajs.shape[0]):
        mean[:,n] = average(trajs[n,:,1:])
    return mean

po = Pool(9)
k_cut_new = np.zeros_like(k)
w_cut_new = np.zeros_like(w)
d_cut_new = np.zeros_like(d)



