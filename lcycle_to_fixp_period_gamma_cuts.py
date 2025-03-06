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

while True:
    if os.path.exists('/home/christian/Documents/Bachelor/numerics/mean_field/4.4k13_1.5w5.5_3d/means_800line_10k_1.7242d1.72403.npy'):
        # sleep(5*60)
        break
    sleep(60*60)
print('start')


root_path = '/home/christian/Documents/Bachelor/numerics/mean_field/gamma_var/'
name = '_k2.5_w0.75_g4'
namenp_gcut = '_g_cut'+name + '.npy'


solnumb_file_gcut = root_path+ f'solnumb' + namenp_gcut
xsol_file_gcut = root_path+f'xsol' +namenp_gcut
ysol_file_gcut = root_path+f'ysol' +namenp_gcut

namenp_wcut = '_w_cut' + name + '.npy'


solnumb_file_wcut = root_path+ f'solnumb' + namenp_wcut
xsol_file_wcut = root_path+f'xsol' +namenp_wcut
ysol_file_wcut = root_path+f'ysol' +namenp_wcut

# params = np.load(root_path+'params71.npy')
# print('worked')
# k = np.load(root_path+ 'kparam' + namenp)
# w = np.load(root_path+ 'wparam' + namenp)
# d = np.load(root_path+ 'dparam' + namenp)

solution_number_gcut = np.load(solnumb_file_gcut)
xsol_gcut = np.load(xsol_file_gcut)
ysol_gcut = np.load(ysol_file_gcut)
gridpoints = int(solution_number_gcut.shape[0])
solution_number_wcut = np.load(solnumb_file_wcut)
xsol_wcut = np.load(xsol_file_wcut)
ysol_wcut = np.load(ysol_file_wcut)


halfgrid = gridpoints//2+gridpoints%2

# gamma = np.linspace(0.1,6,gridpoints)
gamma_cut = 3#gamma[85]
# k= np.linspace(4.4,13,gridpoints)
kappa_cut1 = 3#k[80]
kappa_cut2 =5
# k = np.linspace(14.2,16,gridpoints)
gamma = np.linspace(0,5,gridpoints)
omega_cut = 0.75
w= np.linspace(0.04,2,gridpoints)
d = np.linspace(-6,0,halfgrid)
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

stability_plus_gcut = np.zeros_like(solution_number_gcut)
stability_plus_wcut = np.zeros_like(solution_number_wcut)
stability_gcut = np.zeros_like(xsol_gcut)
stability_wcut = np.zeros_like(xsol_wcut)
# stability_plus = np.load(root_path+'stability_plus.npy')
# stability = np.load(root_path+'stability.npy')
stab_zsol_gcut = np.ones_like(solution_number_gcut)*np.nan
stab_xsol_gcut = np.ones_like(solution_number_gcut)*np.nan
stab_ysol_gcut = np.ones_like(solution_number_gcut)*np.nan

stab_zsol_wcut = np.ones_like(solution_number_wcut)*np.nan
stab_xsol_wcut = np.ones_like(solution_number_wcut)*np.nan
stab_ysol_wcut = np.ones_like(solution_number_wcut)*np.nan

for io in range(gridpoints):
        for id, dval in enumerate(d):
            stability_gcut[io,id], stability_plus_gcut[io,id]=determine_stability(solution_number_gcut[io,id],xsol_gcut[io,id],ysol_gcut[io,id],omega_cut,kappa_cut1,dval,gamma[io])
            stability_wcut[io,id], stability_plus_wcut[io,id]=determine_stability(solution_number_wcut[io,id],xsol_wcut[io,id],ysol_wcut[io,id],w[io],kappa_cut2,dval,gamma_cut)

            if stability_plus_gcut[io,id]:
                # print(kval,wval,dval)
                for n in range(int(solution_number_gcut[io,id])):
                    if stability_gcut[io,id,n]==2:
                        stab_xsol_gcut[io,id]=xsol_gcut[io,id,n]
                        stab_ysol_gcut[io,id] = ysol_gcut[io,id,n]
                        stab_zsol_gcut[io,id] = m_z(xsol_gcut[io,id,n],ysol_gcut[io,id,n],omega_cut,kappa_cut1,Gam)
                        break
            if stability_plus_wcut[io,id]:
                # print(kval,wval,dval)
                for n in range(int(solution_number_wcut[io,id])):
                    if stability_wcut[io,id,n]==2:
                        stab_xsol_wcut[io,id]=xsol_wcut[io,id,n]
                        stab_ysol_wcut[io,id] = ysol_wcut[io,id,n]
                        stab_zsol_wcut[io,id] = m_z(xsol_wcut[io,id,n],ysol_wcut[io,id,n],w[io],kappa_cut2,Gam)
                        break
                



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

start_points_non_prec = starting_grid(0.5,3)
start_points = starting_grid(0.62,4)
# start_points = start_points_non_prec
numb_of_start = int(start_points.shape[0])
numb_of_tsteps = int(10000)



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


def stab_find(wval,kval,dval,sgam):
    arguments = wval,kval, dval, Gam, sgam 
    # start = np.reshape(starting_points(1),3)
    trajs = []
    # start = starting_grid(numb_of_start)
    means = np.zeros((3,numb_of_start))
    averages = np.zeros((3,numb_of_start))
    found_p = np.zeros(numb_of_start)
    for i,s in enumerate(start_points_non_prec):
        traj = solve_ivp(gl2,(0,4000),y0=s,args=arguments,method='DOP853',t_eval=np.linspace(2000,3999,numb_of_tsteps))#np.arange(numb_of_tsteps)/numb_of_tsteps*10000)#,rtol=1e-12,atol=1e-25)
        # traj = solve_ivp(gl2,(0,14000),y0=s,args=arguments,method='DOP853',t_eval=np.linspace(10000,13999,numb_of_tsteps))
        # ind = int(3/4*len(traj.t))
        trajs.append(traj)
        averages[:,i], found_p[i]=average(traj.y)
        
        means[:,i] = np.mean(traj.y,axis=1) 
    # averages = means
    for j in range(start_points_non_prec.shape[0],numb_of_start):
        means[:,j]= means[:,i]
        averages[:,j] = averages[:,i]
    return means, averages, trajs

def stab_find_prec(wval,kval,dval,sgam):
    arguments = wval,kval, dval, Gam, sgam 
    # start = np.reshape(starting_points(1),3)
    trajs = []
    # start = starting_grid(numb_of_start)
    means = np.zeros((3,numb_of_start))
    averages = np.zeros((3,numb_of_start))
    found_p = np.zeros(numb_of_start)
    for i,s in enumerate(start_points):
        # traj = solve_ivp(gl2,(0,1400),y0=s,args=arguments,method='DOP853',t_eval=np.linspace(1000,1399,numb_of_tsteps))#np.arange(numb_of_tsteps)/numb_of_tsteps*10000)#,rtol=1e-12,atol=1e-25)
        traj = solve_ivp(gl2,(0,20000),y0=s,args=arguments,method='DOP853',t_eval=np.linspace(16000,19999,numb_of_tsteps),rtol=1e-10)
        # ind = int(3/4*len(traj.t))
        trajs.append(traj)
        averages[:,i], found_p[i]=average(traj.y)
        
        means[:,i] = np.mean(traj.y,axis=1) 
    # averages = means
    return means, averages, trajs


def count_cycles(m):
    number_c = numb_of_start
    difference = [np.linalg.norm(m[:,0]-m[:,1]),np.linalg.norm(m[:,0]-m[:,2]),np.linalg.norm(m[:,2]-m[:,1])]
    for diff in difference:
        if diff<1/10**4:
            number_c-=1
    return max(number_c,1)

def endp_wrapper(index):
    i , j = index//halfgrid, index%halfgrid
    # if period_found3[i,j]!=1:
        # return
    # w_cut = 72
    # w_cut = 20
    # d_cut = 25
    trajectoriesg=np.ones((numb_of_start,numb_of_tsteps,4))
    trajectoriesk = np.ones_like(trajectoriesg)
    # trajectoriesd = np.ones_like(trajectoriesk)
    
    if stability_plus_wcut[i,j]==0:
        if i<4:
            solk, averk, trajectk = stab_find_prec(w[i],kappa_cut2,d[j],gamma_cut)
        else:
            solk, averk, trajectk = stab_find(w[i],kappa_cut2,d[j],gamma_cut)
        for it in range(len(trajectk)):
            trajectoriesk[it,:,0] = trajectk[it].t
            trajectoriesk[it,:,1:] = trajectk[it].y.transpose()
    else:
        averk = solk = np.row_stack((np.array([stab_xsol_wcut[i,j],stab_ysol_wcut[i,j],stab_zsol_wcut[i,j]]),np.zeros((numb_of_start-1,3)))).transpose()
    if  stability_plus_gcut[i,j]==0:
        if i <6:
            solg, averg ,trajectg = stab_find_prec(omega_cut,kappa_cut1,d[j],gamma[i])
        else:
            solg, averg ,trajectg = stab_find(omega_cut,kappa_cut1,d[j],gamma[i])
        for it in range(len(trajectg)):
            trajectoriesg[it,:,0] = trajectg[it].t
            trajectoriesg[it,:,1:] = trajectg[it].y.transpose()
    else:
        averg = solg = np.row_stack((np.array([stab_xsol_gcut[i,j],stab_ysol_gcut[i,j],stab_zsol_gcut[i,j]]),np.zeros((numb_of_start-1,3)))).transpose()

    # if True:# stability_plus[i,j,d_cut]==0:
    #     sold,trajectd = stab_find(k[i],w[j],d[d_cut])
    #     for it in range(len(trajectd)):
    #         trajectoriesd[it,:,0] = trajectd[it].t
    #         trajectoriesd[it,:,1:] = trajectd[it].y.transpose()
    # else:
        # sold = np.row_stack((np.array([stab_xsol[i,j,d_cut],stab_ysol[i,j,d_cut],stab_zsol[i,j,d_cut]]),np.zeros((numb_of_start-1,3)))).transpose()
    # solk, sold = 0,0
    # return solk, solw, sold, trajectoriesk, trajectoriesw, trajectoriesd#, p_found
    # return solg, averg , trajectoriesg, solk, averk , trajectoriesk
    return solg, averg,  solk, averk 

block_numb = 3

subgridx = gridpoints//block_numb
subgridd = halfgrid//block_numb

g_cutarray = np.zeros((gridpoints,halfgrid,3,numb_of_start))
g_cut_aver = np.zeros_like(g_cutarray)
w_cutarray = np.zeros_like(g_cutarray)
w_cut_aver = np.zeros_like(g_cutarray)



for gridx in range(block_numb):
    for gridy in range(block_numb):
        # w_cut_traj = np.zeros((subgridx,subgridd,numb_of_start,numb_of_tsteps,4))
        # w_cut_traj = np.zeros_like(g_cut_traj)
        po = Pool(9)
        # s = np.zeros((numb_of_tsteps,3))
        # g_cutarray = np.zeros((subgridx,subgridd,3,numb_of_start))
        # g_cut_aver = np.zeros_like(g_cutarray)
        # w_cutarray = np.zeros_like(g_cutarray)
        # w_cut_aver = np.zeros_like(g_cutarray)
        # g_cut_traj = np.zeros((subgridx,subgridd,numb_of_start,numb_of_tsteps,4))
        # w_cut_traj = np.zeros_like(g_cut_traj)
        # d_cut_traj = np.zeros((subgrid,subgrid,numb_of_start,numb_of_tsteps,4))
        # print(g_cut_traj)
        i_mat = np.zeros((subgridx,subgridd),dtype=np.int32)
        for row in range(subgridx):
            i_mat[row,:] = np.arange(subgridd)+subgridd*3*row+3*subgridx*subgridd*gridx+subgridd*gridy
        # period_eval = np.zeros((subgrid,subgrid,numb_of_start))
        result = po.imap(endp_wrapper,np.reshape(i_mat,(subgridx*subgridd)),chunksize=2)
        for index, res in enumerate(result):
            l, m = index//subgridd, index%subgridd
            ind = int(i_mat[l,m])
            i, j = ind//halfgrid, ind%halfgrid
      
            # w_cutarray[i,j], g_cutarray[i,j], d_cutarray[i,j], w_cut_traj[i,j], g_cut_traj[i,j], d_cut_traj[i,j] = res

            # g_cutarray[i,j], g_cut_aver[i,j],trash, w_cutarray[i,j], w_cut_aver[i,j],w_cut_traj[l,m]= res
            g_cutarray[i,j], g_cut_aver[i,j], w_cutarray[i,j], w_cut_aver[i,j]= res

        # print('made it through multiprocessing')
        # for i in range(gridpoints):
        #     for j in  range(gridpoints//2):
        #         w_cut_traj[i,-1-j] = w_cut_traj[i,j]
        #         w_cutarray[i,-1-j] = w_cutarray[i,j]
        #         g_cut_traj[i,-1-j] = g_cut_traj[i,j]
        #         g_cutarray[i,-1-j] = g_cutarray[i,j]
        # np.save(root_path+'w_cutarray_p',w_cutarray)
        # np.save(root_path+f'w_cutarray_p_{gridx}_{gridy}',g_cutarray)
        # # np.save(root_path+f'w_cut_peval_{gridx}_{gridy}',period_eval)
        # np.save(root_path+f'w_cut_aver_p_{gridx}_{gridy}',g_cut_aver)
        # np.save(root_path+'w_cut_period_status3',period_eval)
        # np.save(root_path+'d_cutarray_p',d_cutarray)
        # np.savez_compressed(root_path+f'w_cut_traj_p_{gridx}_{gridy}',w_cut_traj)
        # np.savez_compressed(root_path+f'w_cut_traj_p_{gridx}_{gridy}',g_cut_traj)
        np.save(root_path+f'g_cutarray_p',g_cutarray)
        np.save(root_path+f'g_cut_aver_p',g_cut_aver)
        np.save(root_path+f'w_cutarray_p',w_cutarray)
        np.save(root_path+f'w_cut_aver_p',w_cut_aver)
        del po
        # del w_cut_traj
        # del g_cutarray
        # del g_cut_aver
        # del period_eval
        del result
        gc.collect()
        # np.savez_compressed(root_path+'d_cut_traj_p',d_cut_traj)
        duration = time()-t1
        print("{:.0f}h, {:.0f}min, {:.1f}s, {:d}, {:d}".format(duration//3600, (duration%3600)//60, duration%60, gridx, gridy))
# np.save(root_path+f'g_cutarray_p_{gridx}_{gridy}',g_cutarray)
# np.save(root_path+f'g_cut_aver_p_{gridx}_{gridy}',g_cut_aver)
# np.save(root_path+f'w_cutarray_p_{gridx}_{gridy}',w_cutarray)
# np.save(root_path+f'w_cut_aver_p_{gridx}_{gridy}',w_cut_aver)