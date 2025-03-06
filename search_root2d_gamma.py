import sys
sys.path.append(r'/home/christian/Documents/Bachelor/numerics')
from physical_functions_mf import *
from add_solutions_functions import *
from scipy.optimize import root
from multiprocessing import Pool
from time import time
t1 = time()
root_path = '/home/christian/Documents/Bachelor/numerics/mean_field/gamma_var/'
name = '_k2.5_w0.75_g4'
namenp = name + '.npy'

solnumb_file = root_path+ f'solnumb2' + namenp
xsol_file = root_path+f'xso2' +namenp
ysol_file = root_path+f'ysol2' +namenp

# params = np.load(root_path+'params71.npy')
# print('worked')
# exit()
# k = np.load(root_path+ 'kparam' + namenp)
# w = np.load(root_path+ 'wparam' + namenp)
# d = np.load(root_path+ 'dparam' + namenp)

# solution_number = np.load(solnumb_file)
# xsol = np.load(xsol_file)
# ysol = np.load(ysol_file)
# gridpoints = solution_number.shape[0]
gridpoints = 90
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

xsol_g_cut = np.zeros(2*[gridpoints]+[5])
ysol_g_cut = np.zeros_like(xsol_g_cut)
solution_number_g_cut = np.zeros(2*[gridpoints])

xsol_w_cut = np.zeros(2*[gridpoints]+[5])
ysol_w_cut = np.zeros_like(xsol_w_cut)
solution_number_w_cut = np.zeros(2*[gridpoints])


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
halfgrid = gridpoints//2+gridpoints%2
def wrapper(index):
    # ik, ig, id = index//gridpoints**2,(index%gridpoints**2)//gridpoints, index%gridpoints
    io, id = index//halfgrid, index%halfgrid
    # ik, ig, id = index//(gridpoints*halfgrid),(index%(gridpoints*halfgrid))//halfgrid, index%halfgrid
    # kval, sgam, dval = k[ik], gamma[ig], d[id]
    # if check_sol[ik,iw,id] and dval!=0:
    # return root2d(k[ik],wval,d[id],gamma[ig])
    return root2d(kappa_cut1,omega_cut,d[id],gamma[io]), root2d(kappa_cut2,w[io],d[id],gamma_cut)
    # else:
    #     return [0]
    

po = Pool(9)
result = po.imap(wrapper,range(gridpoints*halfgrid))
# result = po.imap(wrapper,range(gridpoints**3))

reject = np.zeros_like(solution_number_g_cut)
for index, res in enumerate(result):
    # ik, ig, id = index//gridpoints**2,(index%gridpoints**2)//gridpoints, index%gridpoints
    # ik, ig, id = index//(gridpoints*halfgrid),(index%(gridpoints*halfgrid))//halfgrid, index%halfgrid
    # if check_sol[ik,iw,id] and d[id]!=0:
    io, id = index//halfgrid, index%halfgrid
    [solig, xsol_g_cut[io,id,:solig], ysol_g_cut[io,id,:solig]], [solik, xsol_w_cut[io,id,:solik], ysol_w_cut[io,id,:solik]] = res
    
    solution_number_g_cut[io,id] = solig
    solution_number_g_cut[io,-1-id], xsol_g_cut[io,-1-id,:solig], ysol_g_cut[io,-1-id,:solig] = solig,xsol_g_cut[io,id,:solig], ysol_g_cut[io,id,:solig]
    solution_number_w_cut[io,id] = solik
    solution_number_w_cut[io,-1-id], xsol_w_cut[io,-1-id,:solik], ysol_w_cut[io,-1-id,:solik] = solik,xsol_w_cut[io,id,:solik], ysol_w_cut[io,id,:solik]



solnumb_file_g_cut = root_path+ f'solnumb_g_cut' + name
xsol_file_g_cut = root_path+f'xsol_g_cut' +name
ysol_file_g_cut = root_path+f'ysol_g_cut' +name

np.save(solnumb_file_g_cut,solution_number_g_cut)
np.save(xsol_file_g_cut,xsol_g_cut)
np.save(ysol_file_g_cut,ysol_g_cut)
solnumb_file_w_cut = root_path+ f'solnumb_w_cut' + name
xsol_file_w_cut = root_path+f'xsol_w_cut' +name
ysol_file_w_cut = root_path+f'ysol_w_cut' +name

np.save(solnumb_file_w_cut,solution_number_w_cut)
np.save(xsol_file_w_cut,xsol_w_cut)
np.save(ysol_file_w_cut,ysol_w_cut)

print(time()-t1)
