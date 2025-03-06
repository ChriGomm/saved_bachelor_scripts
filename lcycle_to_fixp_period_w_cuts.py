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
name = 'DOP'


namenp_wcut = '_w_cut' + name + '.npy'


solnumb_file_wcut = root_path+ f'solnumb' + namenp_wcut
xsol_file_wcut = root_path+f'xsol' +namenp_wcut
ysol_file_wcut = root_path+f'ysol' +namenp_wcut

# params = np.load(root_path+'params71.npy')
# print('worked')
# k = np.load(root_path+ 'kparam' + namenp)
# w = np.load(root_path+ 'wparam' + namenp)
# d = np.load(root_path+ 'dparam' + namenp)



solution_number_wcut = np.load(solnumb_file_wcut)
xsol_wcut = np.load(xsol_file_wcut)
ysol_wcut = np.load(ysol_file_wcut)
gridpoints = int(solution_number_wcut.shape[0])

# halfgrid = gridpoints//2+gridpoints%2

# gamma = np.linspace(0.1,6,gridpoints)
gamma_cut = 0.2#gamma[85]
# k= np.linspace(4.4,13,gridpoints)

# k = np.linspace(14.2,16,gridpoints)
k = np.linspace(9.751111111111111, 13.0,gridpoints)
d = np.linspace(-3.0, -1.2,gridpoints)
omega_cut = 2.3889
# d = np.linspace(-3,3,gridpoints)



def determine_stability(solnumb,x_sols,y_sols,wval,kval,dval,sgam):
    stability = np.zeros_like(x_sols)
    stability_plus = 0
    for ind in range(int(solnumb)):
        # s = yvariant[ik,ig,id,ind]
        # if s==1:
        #     yofx = y1ofx
        # else:
        #     yofx = y2ofx
        y = y_sols[ind]
        x = x_sols[ind]
        z = m_z(x,y,wval,kval,Gam)
        m = np.array([x,y,z])
        # print(m)
        # jac = jac_gl_np(0,m,wval,kval,dval,Gam,sgam)
        jac = jac2(0,m,wval,kval,dval,Gam,sgam)
        try:
            eigenvalues, vectors =np.linalg.eig(jac)
            # eigi[ik,ig,id,ind,:]=eigenvalues
        except:
            # print(ik,ig,id)
            stability[ind]=-1
            stability_plus=-1
        if np.any(np.real(eigenvalues)>0):
            # if solution_number[ik,ig,id]==3:
                # print(eigenvalues)
            # print(eigenvalues)
            stability[ind]=0
        elif np.any(np.real(eigenvalues==0)):
            stability[ind]=1
            stability_plus=1
        else:
            # pass
            stability[ind] =2
            stability_plus=2
    stability[int(solnumb):]=np.nan
    return stability, stability_plus

stability_plus_wcut = np.zeros_like(solution_number_wcut)
stability_wcut = np.zeros_like(xsol_wcut)
# stability_plus = np.load(root_path+'stability_plus.npy')
# stability = np.load(root_path+'stability.npy')

stab_zsol_wcut = np.ones_like(solution_number_wcut)*np.nan
stab_xsol_wcut = np.ones_like(solution_number_wcut)*np.nan
stab_ysol_wcut = np.ones_like(solution_number_wcut)*np.nan

for io in range(gridpoints):
        for id, dval in enumerate(d):
            stability_wcut[io,id], stability_plus_wcut[io,id]=determine_stability(solution_number_wcut[io,id],xsol_wcut[io,id],ysol_wcut[io,id],omega_cut,k[io],dval,gamma_cut)

            
            if stability_plus_wcut[io,id]:
                # print(kval,wval,dval)
                for n in range(int(solution_number_wcut[io,id])):
                    if stability_wcut[io,id,n]==2:
                        stab_xsol_wcut[io,id]=xsol_wcut[io,id,n]
                        stab_ysol_wcut[io,id] = ysol_wcut[io,id,n]
                        stab_zsol_wcut[io,id] = m_z(xsol_wcut[io,id,n],ysol_wcut[io,id,n],omega_cut,k[io],Gam)
                        break
                
w_cut = int(80)
g_cut = int(5)
d_cut = int(5)
# w_cut = int(72)
# w_cut = int(20)
# d_cut = int(25)

# def draw_border_wcut(i,j,l,w_cut):
#     return (stability_plus[w_cut,max(0,min(gridpoints-1,i+(-1+2*l))),j]+stability_plus[w_cut,i,j]==1) or (stability_plus[w_cut,i,max(0,min(gridpoints-1,j+(-1+2*l)))]+stability_plus[w_cut,i,j]==1)
# def draw_border_wcut(i,j,l,w_cut):
#     return (stability_plus[max(0,min(gridpoints-1,i+(-1+2*l))),w_cut,j]+stability_plus[i,w_cut,j]==1) or (stability_plus[i,w_cut,max(0,min(gridpoints-1,j+(-1+2*l)))]+stability_plus[i,w_cut,j]==1)

# def draw_border_dcut(i,j,l,d_cut):
#     return (stability_plus[max(0,min(gridpoints-1,i+(-1+2*l))),j,d_cut]+stability_plus[i,j,d_cut]==1) or (stability_plus[i,max(0,min(gridpoints-1,j+(-1+2*l))),d_cut]+stability_plus[i,j,d_cut]==1)

# border_wcut = []

# for i in range(gridpoints):
#     for j in range(gridpoints):
#         if draw_border_wcut(i,j,0,w_cut):
#             border_wcut.append([w[i],d[j]])
#         if draw_border_wcut(i,j,0,w_cut):
#             border_wcut.append([w[i],d[j]])
#         if draw_border_wcut(i,j,0,w_cut):
#             border_wcut.append([w[i],d[j]])




# def starting_points(numb):
#     points = np.zeros((numb,3))
#     for i in range(numb):
#         phi = random()*2*np.pi
#         theta = random()*np.pi
#         r = random()*0.5
#         points[i,:] = r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta)
#     return points

def starting_grid(resolution):
    points = np.zeros((resolution,resolution,resolution,3))
    flat = []
    steps = 1/(resolution-1)
    for i in range(resolution):
        for j in range(resolution):
            for l in range(resolution):
                points[i,j,l]= -0.5+np.array([i,j,l])*steps
                if np.linalg.norm(points[i,j,l])>0.5:
                    points[i,j,l]=np.nan
                else:
                    flat.append(points[i,j,l])
    return np.row_stack(flat)

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

start_points = starting_grid(5)
numb_of_start = int(start_points.shape[0])
numb_of_tsteps = int(10000)


def stab_find(wval,kval,dval,sgam):
    arguments = wval,kval, dval, Gam, sgam 
    # start = np.reshape(starting_points(1),3)
    trajs = []
    # start = starting_grid(numb_of_start)
    means = np.zeros((3,numb_of_start))
    averages = np.zeros((3,numb_of_start))
    found_p = np.zeros(numb_of_start)
    for i,s in enumerate(start_points):
        # traj = solve_ivp(gl2,(0,1400),y0=s,args=arguments,method='DOP853',t_eval=np.linspace(1000,1390,numb_of_tsteps))#np.arange(numb_of_tsteps)/numb_of_tsteps*10000)#,rtol=1e-12,atol=1e-25)
        traj = solve_ivp(gl2,(0,14000),y0=s,args=arguments,method='DOP853',t_eval=np.linspace(10000,13999,numb_of_tsteps))
        # ind = int(3/4*len(traj.t))
        trajs.append(traj)
        # averages[:,i], found_p[i]=average(traj.y)
        
        means[:,i] = np.mean(traj.y,axis=1) 
    averages = means
    return means, averages, trajs




def count_cycles(m):
    number_c = numb_of_start
    difference = [np.linalg.norm(m[:,0]-m[:,1]),np.linalg.norm(m[:,0]-m[:,2]),np.linalg.norm(m[:,2]-m[:,1])]
    for diff in difference:
        if diff<1/10**4:
            number_c-=1
    return max(number_c,1)

def endp_wrapper(index):
    i , j = index//gridpoints, index%gridpoints
    # if period_found3[i,j]!=1:
        # return
    # w_cut = 72
    # w_cut = 20
    # d_cut = 25
    trajectoriesg=np.ones((numb_of_start,numb_of_tsteps,4))
    trajectoriesk = np.ones_like(trajectoriesg)
    # trajectoriesd = np.ones_like(trajectoriesk)
    
    if stability_plus_wcut[i,j]==0:
        solk, averk, trajectk = stab_find(omega_cut,k[i],d[j],gamma_cut)
        for it in range(len(trajectk)):
            trajectoriesk[it,:,0] = trajectk[it].t
            trajectoriesk[it,:,1:] = trajectk[it].y.transpose()
    else:
        averk = solk = np.row_stack([np.array([stab_xsol_wcut[i,j],stab_ysol_wcut[i,j],stab_zsol_wcut[i,j]])]*numb_of_start).transpose()
    # if  stability_plus_gcut[i,j]==0:
    #     solg, averg ,trajectg = stab_find(w[i],kappa_cut,gamma_cut,d[j])
    #     for it in range(len(trajectg)):
    #         trajectoriesg[it,:,0] = trajectg[it].t
    #         trajectoriesg[it,:,1:] = trajectg[it].y.transpose()
    # else:
    #     averg = solg = np.row_stack((np.array([stab_xsol_gcut[i,j],stab_ysol_gcut[i,j],stab_zsol_gcut[i,j]]),np.zeros((numb_of_start-1,3)))).transpose()

    # if True:# stability_plus[i,j,d_cut]==0:
    #     sold,trajectd = stab_find(k[i],w[j],d[d_cut])
    #     for it in range(len(trajectd)):
    #         trajectoriesd[it,:,0] = trajectd[it].t
    #         trajectoriesd[it,:,1:] = trajectd[it].y.transpose()
    # else:
        # sold = np.row_stack((np.array([stab_xsol[i,j,d_cut],stab_ysol[i,j,d_cut],stab_zsol[i,j,d_cut]]),np.zeros((numb_of_start-1,3)))).transpose()
    solk, sold = 0,0
    # return solk, solw, sold, trajectoriesk, trajectoriesw, trajectoriesd#, p_found
    # return solg, averg , trajectoriesg, solk, averk , trajectoriesk
    return  solk, averk 

# block_numb = 3

# subgridx = gridpoints//block_numb
# subgridd = gridpoints//block_numb

# g_cutarray = np.zeros((gridpoints,gridpoints,3,numb_of_start))#
# g_cut_aver = np.zeros_like(g_cutarray)
# w_cutarray = np.zeros_like(g_cutarray)
# w_cut_aver = np.zeros_like(g_cutarray)



# for gridx in range(block_numb):
#     for gridy in range(block_numb):
#         # w_cut_traj = np.zeros((subgridx,subgridd,numb_of_start,numb_of_tsteps,4))
#         # w_cut_traj = np.zeros_like(g_cut_traj)
#         po = Pool(9)
#         # s = np.zeros((numb_of_tsteps,3))
#         # g_cutarray = np.zeros((subgridx,subgridd,3,numb_of_start))
#         # g_cut_aver = np.zeros_like(g_cutarray)
#         # w_cutarray = np.zeros_like(g_cutarray)
#         # w_cut_aver = np.zeros_like(g_cutarray)
#         # g_cut_traj = np.zeros((subgridx,subgridd,numb_of_start,numb_of_tsteps,4))
#         # w_cut_traj = np.zeros_like(g_cut_traj)
#         # d_cut_traj = np.zeros((subgrid,subgrid,numb_of_start,numb_of_tsteps,4))
#         # print(g_cut_traj)
#         i_mat = np.zeros((subgridx,subgridd),dtype=np.int32)
#         for row in range(subgridx):
#             i_mat[row,:] = np.arange(subgridd)+subgridd*3*row+3*subgridx*subgridd*gridx+subgridd*gridy
#         # period_eval = np.zeros((subgrid,subgrid,numb_of_start))
#         result = po.imap(endp_wrapper,np.reshape(i_mat,(subgridx*subgridd)),chunksize=2)
#         for index, res in enumerate(result):
#             l, m = index//subgridd, index%subgridd
#             ind = int(i_mat[l,m])
#             i, j = ind//gridpoints, ind%gridpoints
      
#             # w_cutarray[i,j], g_cutarray[i,j], d_cutarray[i,j], w_cut_traj[i,j], g_cut_traj[i,j], d_cut_traj[i,j] = res

#             # g_cutarray[i,j], g_cut_aver[i,j],trash, w_cutarray[i,j], w_cut_aver[i,j],w_cut_traj[l,m]= res
#             w_cutarray[i,j], w_cut_aver[i,j]= res

#         # print('made it through multiprocessing')
#         # for i in range(gridpoints):
#         #     for j in  range(gridpoints//2):
#         #         w_cut_traj[i,-1-j] = w_cut_traj[i,j]
#         #         w_cutarray[i,-1-j] = w_cutarray[i,j]
#         #         g_cut_traj[i,-1-j] = g_cut_traj[i,j]
#         #         g_cutarray[i,-1-j] = g_cutarray[i,j]
#         # np.save(root_path+'w_cutarray_p',w_cutarray)
#         # np.save(root_path+f'w_cutarray_p_{gridx}_{gridy}',g_cutarray)
#         # # np.save(root_path+f'w_cut_peval_{gridx}_{gridy}',period_eval)
#         # np.save(root_path+f'w_cut_aver_p_{gridx}_{gridy}',g_cut_aver)
#         # np.save(root_path+'w_cut_period_status3',period_eval)
#         # np.save(root_path+'d_cutarray_p',d_cutarray)
#         # np.savez_compressed(root_path+f'w_cut_traj_p_{gridx}_{gridy}',w_cut_traj)
#         # np.savez_compressed(root_path+f'w_cut_traj_p_{gridx}_{gridy}',g_cut_traj)

#         np.save(root_path+name+'_w_cutarray_p',w_cutarray)
#         np.save(root_path+name+'_w_cut_aver_p',w_cut_aver)
#         del po
#         # del w_cut_traj
#         # del g_cutarray
#         # del g_cut_aver
#         # del period_eval
#         del result
#         gc.collect()
#         # np.savez_compressed(root_path+'d_cut_traj_p',d_cut_traj)
#         duration = time()-t1
#         print("{:.0f}h, {:.0f}min, {:.1f}s, {:d}, {:d}".format(duration//3600, (duration%3600)//60, duration%60, gridx, gridy))
