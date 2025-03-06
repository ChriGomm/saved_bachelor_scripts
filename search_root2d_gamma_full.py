import sys
sys.path.append(r'/home/christian/Documents/Bachelor/numerics')
from physical_functions_mf import *
from add_solutions_functions import *
from scipy.optimize import root
from multiprocessing import Pool
from time import time
t1 = time()
root_path = '/home/christian/Documents/Bachelor/numerics/mean_field/gamma_var/'
name = '_4.4k13_0.2g6_3d_'
namenp = name + '.npy'

gridpoints = 91
halfgrid = gridpoints//2+gridpoints%2
gamma = np.linspace(0.1,6,gridpoints)
gamma_cut = 6#gamma[85]
k= np.linspace(4.4,13,gridpoints)
kappa_cut = 17#k[80]
# k = np.linspace(14,18,gridpoints)
# gamma = np.linspace(4,10,gridpoints)
wval = 0.75
d = np.linspace(-4,0,halfgrid)
# d = np.linspace(-3,3,gridpoints)

xsol_g_cut = np.zeros(3*[gridpoints]+[5])
ysol_g_cut = np.zeros_like(xsol_g_cut)
solution_number_g_cut = np.zeros(3*[gridpoints])




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


halfgrid = gridpoints//2+gridpoints%2
def wrapper(index):
    # ik, ig, id = index//gridpoints**2,(index%gridpoints**2)//gridpoints, index%gridpoints
    # io, id = index//halfgrid, index%halfgrid
    ik, ig, id = index//(gridpoints*halfgrid),(index%(gridpoints*halfgrid))//halfgrid, index%halfgrid
    # kval, sgam, dval = k[ik], gamma[ig], d[id]
    # if check_sol[ik,iw,id] and dval!=0:
    return root2d(k[ik],wval,d[id],gamma[ig])
    # return root2d(kappa_cut,wval,d[id],gamma[io]), root2d(k[io],wval,d[id],gamma_cut)
    # else:
    #     return [0]
    

po = Pool(9)
result = po.imap(wrapper,range(gridpoints**2*halfgrid))
# result = po.imap(wrapper,range(gridpoints**3))

reject = np.zeros_like(solution_number_g_cut)
for index, res in enumerate(result):
    # ik, ig, id = index//gridpoints**2,(index%gridpoints**2)//gridpoints, index%gridpoints
    ik, ig, id = index//(gridpoints*halfgrid),(index%(gridpoints*halfgrid))//halfgrid, index%halfgrid
    # if check_sol[ik,iw,id] and d[id]!=0:
    # io, id = index//halfgrid, index%halfgrid
    solig, xsol_g_cut[ik,ig,id,:solig], ysol_g_cut[ik,ig,id,:solig]= res
    
    solution_number_g_cut[ik,ig,id] = solig
    solution_number_g_cut[ik,ig,-1-id], xsol_g_cut[ik,ig,-1-id,:solig], ysol_g_cut[ik,ig,-1-id,:solig] = solig,xsol_g_cut[ik,ig,id,:solig], ysol_g_cut[ik,ig,id,:solig]



solnumb_file = root_path+ f'solnumb' + name
xsol_file = root_path+f'xsol' +name
ysol_file = root_path+f'ysol' +name

np.save(solnumb_file,solution_number_g_cut)
np.save(xsol_file,xsol_g_cut)
np.save(ysol_file,ysol_g_cut)

print(time()-t1)
