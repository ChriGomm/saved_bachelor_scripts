import sys
sys.path.append(r'/home/christian/Documents/Bachelor/numerics')
from physical_functions_mf import *
from add_solutions_functions import *
from scipy.optimize import root
from multiprocessing import Pool
from time import time
from random import random
import gc
t1 = time()
root_path = '/home/christian/Documents/Bachelor/numerics/mean_field/4.4k13_1.5w5.5_3d/'
name = '_4.4k13_1.5w5.5'
namenp = name + '.npy'

solnumb_file = root_path+ f'solnumb' + namenp
xsol_file = root_path+f'xsol' +namenp
ysol_file = root_path+f'ysol' +namenp

# params = np.load(root_path+'params71.npy')
# print('worked')
# k = np.load(root_path+ 'kparam' + namenp)
# w = np.load(root_path+ 'wparam' + namenp)
# d = np.load(root_path+ 'dparam' + namenp)

gridpoints = 54

k = np.linspace(9.751111111111111, 13.0,gridpoints)
d = np.linspace(-3.0, -1.2,gridpoints)
w = 2.3889

gridpoints = 54
w_cut_array_old = np.load(root_path+'w_cut_aver_corrected.npy')
number_c = np.load(root_path+'w_cut_number_of_lc_corrected.npy')
# w_cutarray3 = np.zeros((gridpoints,gridpoints,3,15))
# period_found3 = np.zeros((gridpoints,gridpoints,15))
# subgrid = gridpoints//3
# for i in range(3):
#     for j in range(3):
#         w_cutarray3[i*subgrid:(i+1)*subgrid,j*subgrid:(j+1)*subgrid,:,:] = np.load(root_path+f'w_cut_aver_p_{i}_{j}.npy')
#         period_found3[i*subgrid:(i+1)*subgrid,j*subgrid:(j+1)*subgrid,:] = np.load(root_path+f'w_cut_peval_{i}_{j}.npy')

# solution_number = np.load(solnumb_file)
# xsol = np.load(xsol_file)
# ysol = np.load(ysol_file)
# gridpoints = solution_number.shape[0]

# stability_plus = np.load(root_path+'stability_plus.npy')
# stability = np.load(root_path+'stability.npy')
# stab_zsol = np.ones_like(solution_number)*np.nan
# stab_xsol = np.ones_like(solution_number)*np.nan
# stab_ysol = np.ones_like(solution_number)*np.nan
# for ik, kval in enumerate(k):
#     for iw, wval in enumerate(w):
#         for id, dval in enumerate(d):
#             if stability_plus[ik,iw,id]:
#                 # print(kval,wval,dval)
#                 for n in range(int(solution_number[ik,iw,id])):
#                     if stability[ik,iw,id,n]==2:
#                         stab_xsol[ik,iw,id]=xsol[ik,iw,id,n]
#                         stab_ysol[ik,iw,id] = ysol[ik,iw,id,n]
#                         stab_zsol[ik,iw,id] = m_z(xsol[ik,iw,id,n],ysol[ik,iw,id,n],wval,kval,Gam)
#                         break
                
# k_cut = int(72)
# w_cut = int(20)
# d_cut = int(25)

# def draw_border_kcut(i,j,l,k_cut):
#     return (stability_plus[k_cut,max(0,min(gridpoints-1,i+(-1+2*l))),j]+stability_plus[k_cut,i,j]==1) or (stability_plus[k_cut,i,max(0,min(gridpoints-1,j+(-1+2*l)))]+stability_plus[k_cut,i,j]==1)
# def draw_border_wcut(i,j,l,w_cut):
#     return (stability_plus[max(0,min(gridpoints-1,i+(-1+2*l))),w_cut,j]+stability_plus[i,w_cut,j]==1) or (stability_plus[i,w_cut,max(0,min(gridpoints-1,j+(-1+2*l)))]+stability_plus[i,w_cut,j]==1)

# def draw_border_dcut(i,j,l,d_cut):
#     return (stability_plus[max(0,min(gridpoints-1,i+(-1+2*l))),j,d_cut]+stability_plus[i,j,d_cut]==1) or (stability_plus[i,max(0,min(gridpoints-1,j+(-1+2*l))),d_cut]+stability_plus[i,j,d_cut]==1)

# border_kcut = []

# for i in range(gridpoints):
#     for j in range(gridpoints):
#         if draw_border_kcut(i,j,0,k_cut):
#             border_kcut.append([w[i],d[j]])
#         if draw_border_kcut(i,j,0,k_cut):
#             border_kcut.append([w[i],d[j]])
#         if draw_border_kcut(i,j,0,k_cut):
#             border_kcut.append([w[i],d[j]])

numb_of_start = int(30)
mgrid = 5
numb_of_tsteps = int(10000)

# def starting_points(numb):
#     points = np.zeros((numb,3))
#     for i in range(numb):
#         phi = random()*2*np.pi
#         theta = random()*np.pi
#         r = random()*0.5
#         points[i,:] = r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta)
#     return points

# def starting_points(gridr,gridthet,gridphi):
#     points = np.zeros((gridr,gridthet,gridphi,3))
#     r_steps = 0.5/gridr
#     phi_steps = 2*np.pi/gridthet
#     theta_steps = np.pi/gridphi
#     for i in range(gridr):
#         for j in range(gridthet):
#             for l in range(gridphi):
#                 phi = random()*phi_steps+j*phi_steps
#                 theta = random()*theta_steps+l*theta_steps
#                 r = (i+random())*r_steps
#                 points[i,j,l,:] = r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta)
#     return np.reshape(points,(gridr*gridthet*gridphi,3))

def start_slice(r_numb):
    r_steps = 0.5/r_numb
    points = np.zeros((int((r_numb+1)*r_numb/2),2))
    for i in range(r_numb):
        phi_steps = np.pi*2/(i+1)
        r = (i*r_steps+0.5*r_steps)
        offset = random()*np.pi/4*3
        for j in range(i+1):
            phi = (j+0.5)*phi_steps
            points[int((i+1)*i/2)+j,:]= r*np.cos(phi+offset), r*np.sin(phi+offset)
    return points

def starting_points(r_numb):
    slice1 = start_slice(r_numb)
    slice2 = start_slice(r_numb)
    points = np.zeros((2*slice1.shape[0],3))
    points[:slice1.shape[0],:2]=slice1
    points[slice1.shape[0]:,1:] = slice2
    return points
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
            start +=200
            
        else:
            pfound =1
            break
    #     pass
    #     # print('error no period found')
    # else:
    #     # print('period found')
    # print('found period')
    return np.mean(traj[:,start:j],axis=1), pfound

def stab_find(kval,wval,dval):
    arguments = wval,kval, dval, Gam, sgam 
    # start = np.reshape(starting_points(1),3)
    trajs = []
    start = starting_points(mgrid)
    means = np.zeros((3,numb_of_start))
    averages = np.zeros((3,numb_of_start))
    found_p = np.zeros(numb_of_start)
    for i,s in enumerate(start):
        # traj = solve_ivp(gl2,(0,140),y0=s,args=arguments,method='LSODA',t_eval=np.linspace(100,139,numb_of_tsteps))#np.arange(numb_of_tsteps)/numb_of_tsteps*10000)#,rtol=1e-12,atol=1e-25)
        traj = solve_ivp(gl2,(0,14000),y0=s,args=arguments,method='LSODA',t_eval=np.linspace(10000,13999,numb_of_tsteps))
        # ind = int(3/4*len(traj.t))
        trajs.append(traj)
        averages[:,i], found_p[i]=average(traj.y)
        # means[:,i] = np.mean(traj.y,axis=1) 
    return means, averages, trajs, found_p


# def area_select(i,j):
#     i , j = int(i), int(j)
#     l = number_c.shape[0]
#     for n in range(4):
#         if number_c[min(int(i+n),l-1),j]>1 or number_c[max(0,int(i-n)),j]>1:
#             return True
#     for n in range(4):
#         if number_c[i,min(int(j+n),l-1)]>1 or number_c[i,max(0,int(j-n))]>1:
#             return True
#     return False

def area_select(i,j):
    i , j = int(i), int(j)
    l = number_c.shape[0]
    hit_count = 0
    for n in range(2):
        if number_c[min(int(i+n),l-1),j]==1 or number_c[max(0,int(i-n)),j]==1:
            hit_count+=1
        if number_c[min(int(i+n),l-1),j]>1 or number_c[max(0,int(i-n)),j]>1:
            hit_count-=1
    # if hit_count>1:
    #     return True
    # hit_count = 0
    for n in range(2):
        if number_c[i,min(int(j+n),l-1)]==1 or number_c[i,max(0,int(j-n))]==1:
            hit_count+=1
        if number_c[i,min(int(j+n),l-1)]>1 or number_c[i,max(0,int(j-n))]>1:
            hit_count-=1
    if np.abs(hit_count)<3:
        return True
    return False


# def count_cycles(m):
#     number_c = numb_of_start
#     difference = [np.linalg.norm(m[:,0]-m[:,1]),np.linalg.norm(m[:,0]-m[:,2]),np.linalg.norm(m[:,2]-m[:,1])]
#     for diff in difference:
#         if diff<1/10**4:
#             number_c-=1
#     return max(number_c,1)

def endp_wrapper(index):
    index = int(index)
    i , j = index//gridpoints, index%gridpoints
    if not area_select(i,j):
        return
    # k_cut = 72
    # w_cut = 20
    # d_cut = 25
    trajectoriesw=np.ones((numb_of_start,numb_of_tsteps,4))
    # trajectoriesk = np.ones_like(trajectoriesw)
    # trajectoriesd = np.ones_like(trajectoriesk)
    
    # if True:#stability_plus[k_cut,i,j]==0 and w[i]<=2.73 and d[j]<0:
    #     solk,trajectk = stab_find(k[k_cut],w[i],d[j])
    #     for it in range(len(trajectk)):
    #         trajectoriesk[it,:,0] = trajectk[it].t
    #         trajectoriesk[it,:,1:] = trajectk[it].y.transpose()
    # else:
    #     solk = np.row_stack((np.array([stab_xsol[k_cut,i,j],stab_ysol[k_cut,i,j],stab_zsol[k_cut,i,j]]),np.zeros((numb_of_start-1,3)))).transpose()
    if True:# stability_plus[i,w_cut,j]==0 and k[i]>9.4 and d[j]<0:
        solw, aver ,trajectw, p_found= stab_find(k[i],w,d[j])
        for it in range(len(trajectw)):
            trajectoriesw[it,:,0] = trajectw[it].t
            trajectoriesw[it,:,1:] = trajectw[it].y.transpose()
    else:
        solw = np.row_stack((np.array([stab_xsol[i,w_cut,j],stab_ysol[i,w_cut,j],stab_zsol[i,w_cut,j]]),np.zeros((numb_of_start-1,3)))).transpose()
    # if True:# stability_plus[i,j,d_cut]==0:
    #     sold,trajectd = stab_find(k[i],w[j],d[d_cut])
    #     for it in range(len(trajectd)):
    #         trajectoriesd[it,:,0] = trajectd[it].t
    #         trajectoriesd[it,:,1:] = trajectd[it].y.transpose()
    # else:
        sold = np.row_stack((np.array([stab_xsol[i,j,d_cut],stab_ysol[i,j,d_cut],stab_zsol[i,j,d_cut]]),np.zeros((numb_of_start-1,3)))).transpose()
    solk, sold = 0,0
    # return solk, solw, sold, trajectoriesk, trajectoriesw, trajectoriesd#, p_found
    return aver #, trajectoriesw

subgrid = gridpoints
# w_cutarray = np.zeros((gridpoints,gridpoints,3,numb_of_start))
w_cut_aver = np.zeros((gridpoints,gridpoints,3,numb_of_start))
# w_cutarray[:,:,:,:15] = w_cut_array_old
w_cut_aver[:,:,:,:27] = w_cut_array_old
# w_cut_aver = w_cut_array_old[:,:,:,:8]
# for gridx in range(2,3):
#     # w_cut_traj = np.zeros((subgrid,gridpoints,numb_of_start,numb_of_tsteps,4))
#     # if gridx!=1:
#     #     w_cut_traj[:,:,:15,:,:]  =np.load(root_path+f'w_traj_period_row_{gridx}.npz')['arr_0']
#     for gridy in range(3):
        
po = Pool(9)
# s = np.zeros((numb_of_tsteps,3))
# w_cutarray = np.zeros((subgrid,subgrid,3,numb_of_start))
# w_cut_aver = np.zeros_like(w_cutarray)
# k_cutarray = np.zeros_like(k_cutarray)
# d_cutarray = np.zeros_like(k_cutarray)
# k_cut_traj = np.zeros((subgrid,subgrid,numb_of_start,numb_of_tsteps,4))
# w_cut_traj = np.zeros((subgrid,subgrid,numb_of_start,numb_of_tsteps,4))
# d_cut_traj = np.zeros((subgrid,subgrid,numb_of_start,numb_of_tsteps,4))
# print(w_cut_traj)
# i_mat = np.zeros((subgrid,subgrid))
# for row in range(subgrid):
#     i_mat[row,:] = np.arange(subgrid)+subgrid*3*row+3*subgrid**2*gridx+subgrid*gridy
# period_eval = np.zeros((subgrid,subgrid,numb_of_start))
result = po.imap(endp_wrapper,np.arange(gridpoints**2),chunksize=10)
for index, res in enumerate(result):
    i, j = index//subgrid, index%subgrid
    # ind = int(i_mat[l,m])
    # i, j = ind//gridpoints, ind%gridpoints
    # k_cutarray[i,j], w_cutarray[i,j], d_cutarray[i,j], k_cut_traj[i,j], w_cut_traj[i,j], d_cut_traj[i,j] = res
    if area_select(i,j):
        # w_cut_aver[i,j],w_cut_traj[i-subgrid*gridx,j] = res
        w_cut_aver[i,j] = res

print('made it through multiprocessing')
        # for i in range(gridpoints):
        #     for j in  range(gridpoints//2):
        #         k_cut_traj[i,-1-j] = k_cut_traj[i,j]
        #         k_cutarray[i,-1-j] = k_cutarray[i,j]
        #         w_cut_traj[i,-1-j] = w_cut_traj[i,j]
        #         w_cutarray[i,-1-j] = w_cutarray[i,j]
        # np.save(root_path+'k_cutarray_p',k_cutarray)
        # np.save(root_path+f'w_cutarray_p_{gridx}_{gridy}',w_cutarray)
        # np.save(root_path+f'w_cut_peval_{gridx}_{gridy}',period_eval)
        # np.save(root_path+f'w_cut_aver_p_{gridx}_{gridy}',w_cut_aver)
        # np.save(root_path+'w_cut_period_status3',period_eval)
        # np.save(root_path+'d_cutarray_p',d_cutarray)
        # np.savez_compressed(root_path+'k_cut_traj_p',k_cut_traj)
    # np.savez_compressed(root_path+f'w_cut_traj_p_corrected_row{gridx}',w_cut_traj)
    
    # del w_cut_traj
    # gc.collect()
        # del w_cutarray
        # del w_cut_aver
        # del period_eval
    
    
        # np.savez_compressed(root_path+'d_cut_traj_p',d_cut_traj)
duration = time()-t1
print(f"{duration//3600}h, {(duration%3600)//60}min, {duration%60}s")

# np.save(root_path+f'w_cut_peval_corrected',period_eval)
np.save(root_path+f'w_cut_aver_p_corrected4',w_cut_aver)
print('finished')