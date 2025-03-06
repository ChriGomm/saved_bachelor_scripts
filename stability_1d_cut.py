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

gridpoints = 500
# kappa_cut = 10.118909853249477
# k = 10.60930817610063 
kappa_cut = 11.896603773584905
# d= np.linspace(-1.85, -1.66,gridpoints)
# d= np.linspace(-2.2,-1.63,gridpoints)
# d= np.linspace(-2.3,-2.11,gridpoints)
# d= np.linspace(-1.725,-1.7235,gridpoints)
# d = np.linspace(-1.7242,-1.72403,gridpoints)
# d = np.linspace(-2.798, -2.74,gridpoints)
d = np.linspace(-2.57, -1,gridpoints)
omega_cut = 2.3889
gamma_cut = 0.2
# d = np.linspace(-3,3,gridpoints)

xsol_gcut = np.zeros([gridpoints]+[5])
ysol_gcut = np.zeros_like(xsol_gcut)
solution_number_gcut = np.zeros([gridpoints])




mgridpoints = 40
mgrid = np.zeros((mgridpoints,mgridpoints,2))
rho = np.linspace(0,0.5,mgridpoints)
phi = np.linspace(0,2*np.pi,mgridpoints)
for i, dist in enumerate(rho):
    for j, ang in enumerate(phi):
        mgrid[i,j,:]= dist*np.cos(ang), dist*np.sin(ang)

def root2d(k,w,d,sgam):
    solnumb =0
    rejections=0
    xsol_list =[]
    ysol_list = []
    rejected = []
    for i in range(mgrid.shape[0]):
        for j in range(mgrid.shape[1]):
            r = root(Foriginal_vec,args=(w,k,d,Gam,sgam),x0=mgrid[i,j])
            xsol_list, ysol_list, solnumb, rejections, rejected = add_sol2d(k,w,d,sgam,r,xsol_list,ysol_list,solnumb,rejections,rejected)
    return solnumb, xsol_list, ysol_list#, rejections

# check_sol = np.zeros_like(solution_number)
# for ik, kval in enumerate(k):
#     for iw, wval in enumerate(w):
#         for id, dval in enumerate(d):
#             for i in range(-2,3):
#                 if solution_number[ik,iw,(id+i)%gridpoints]:
#                     check_sol[ik,iw,id]=1
#                 elif solution_number[ik,(iw+i)%gridpoints,id]:
#                     check_sol[ik,iw,id]=1
#                 elif solution_number[(ik+i)%gridpoints,iw,id]:
#                     check_sol[ik,iw,id]=1

def wrapper(index):

    return root2d(kappa_cut,omega_cut,d[index],gamma_cut)
    #     return [0]
    

po = Pool(9)
result = po.imap(wrapper,range(gridpoints),chunksize=20)
# result = po.imap(wrapper,range(gridpoints**3))

reject = np.zeros_like(solution_number_gcut)
for index, res in enumerate(result):

    solig, xsol_gcut[index,:solig], ysol_gcut[index,:solig] = res
    
    solution_number_gcut[index] = solig
    



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
stability_gcut = np.zeros_like(xsol_gcut)
# stability_plus = np.load(root_path+'stability_plus.npy')
# stability = np.load(root_path+'stability.npy')
stab_zsol_gcut = np.ones_like(solution_number_gcut)*np.nan
stab_xsol_gcut = np.ones_like(solution_number_gcut)*np.nan
stab_ysol_gcut = np.ones_like(solution_number_gcut)*np.nan

for id, dval in enumerate(d):
    stability_gcut[id], stability_plus_gcut[id]=determine_stability(solution_number_gcut[id],xsol_gcut[id],ysol_gcut[id],omega_cut,kappa_cut,dval,gamma_cut)
    

    if stability_plus_gcut[id]:
        
        for n in range(int(solution_number_gcut[id])):
            if stability_gcut[id,n]==2:
                stab_xsol_gcut[id]=xsol_gcut[id,n]
                stab_ysol_gcut[id] = ysol_gcut[id,n]
                stab_zsol_gcut[id] = m_z(xsol_gcut[id,n],ysol_gcut[id,n],omega_cut,kappa_cut,Gam)
                break
    if stability_plus_gcut[id]>0 and stability_plus_gcut[id-1]==0 and id!=0:
        print(dval,stab_xsol_gcut[id])
root_path = '/home/christian/Documents/Bachelor/numerics/mean_field/4.4k13_1.5w5.5_3d/'

np.save(root_path+'stab_xsol_dline_bay',stab_xsol_gcut)

duration = time()-t1
print("{:.0f}h, {:.0f}min, {:.1f}s".format(duration//3600, (duration%3600)//60, duration%60))







