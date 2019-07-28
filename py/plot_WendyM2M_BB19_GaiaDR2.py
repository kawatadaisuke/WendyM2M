
#
# WendyM2M for BB19 Gaia DR2 data
#
# Read ../BB19_GaiaDR2/py/gdr2_bb18_*.h5 and apply WendyM2M.
#


import os
import numpy
import h5py
from scipy.misc import logsumexp
# import tqdm
import pickle
import hom2m
import wendypy
import wendym2m
from galpy.util import bovy_plot, save_pickles
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import FuncFormatter
numpy.random.seed(4)
import copy
from matplotlib import gridspec
import seaborn as sns
from scipy.optimize import curve_fit
tc= sns.color_palette('colorblind')
init_color= sns.color_palette()[0]
final_color= tc[2]
constraint_color= tc[1]
save_figures= True
save_chain_figures= False
_SAVE_GIFS= False
plt.rcParams.update({'axes.labelsize': 17.,
              'font.size': 12.,
              'legend.fontsize': 17.,
              'xtick.labelsize':15.,
              'ytick.labelsize':15.,
              'text.usetex': _SAVE_GIFS,
              'figure.figsize': [5,5],
              'xtick.major.size' : 4,
              'ytick.major.size' : 4,
              'xtick.minor.size' : 2,
              'ytick.minor.size' : 2,
              'legend.numpoints':1})
import copy
# for not displaying
matplotlib.use('Agg')
numpy.random.seed(2)

# read the vertical number density data
# colour ID in BB19_GaiaDR2
cid = 0
file_posinput='../BB19_GaiaDR2/py/gdr2_bb18_pos_colid'+str(cid)+'.h5'
with h5py.File(file_posinput, 'r') as f:
    print(f.keys())
    z_pmock = f['z'].value
print(' Number of stars for vertical number density =',len(z_pmock))

file_posinput='../BB19_GaiaDR2/py/gdr2_bb18_posvz_colid'+str(cid)+'.h5'
with h5py.File(file_posinput, 'r') as f:
    z_vmock = f['z'].value
    vz_vmock = f['vz'].value
print(' Number of stars for vertical velocity =',len(z_vmock))

###  Set solar vertical position and velocity

# tempolary set xnm_true
xnm_true = 1.0/(250.0**2)
# vertical positon of the Sun from BB19
zsun_true = 0.0208
# vertical solar velocity
vzsun_true =  7.25
# adjust the vertical positons and velocities.
z_pmock = z_pmock+zsun_true
z_vmock = z_vmock+zsun_true
vz_vmock = vz_vmock+vzsun_true
# set mass =1
m_pmock = numpy.ones_like(z_pmock)
m_vmock = numpy.ones_like(z_vmock)

### Unit normalisation: 2 pi G= L (kpc) = V (km/s) = 1

# units
twopigunit_cgs = (2.0*numpy.pi)*6.67408e-8
msun_cgs = 1.98847e33 # 1 Msun
lunit_kpc = 1.0
vunit_kms = 1.0
lunit_pc = lunit_kpc*1000.0
lunit_cgs = lunit_kpc*3.0857e21 # 1 kpc in cm
vunit_cgs = vunit_kms*1.0e5 # 1 km/s in cm/s
# time unit
tunit_cgs = lunit_cgs/vunit_cgs
print(' time unit (s) =', tunit_cgs)
yr_s = ((365.0*24.0*3600.0)*3+(366.0*24.0*3600.0))/4.0
print(' 1 year (s)=', yr_s)
print(' time unit (Myr) = ',tunit_cgs/yr_s/1.0e6)
# mass unit 
munit_cgs = lunit_cgs**3/(twopigunit_cgs*(tunit_cgs**2))
munit_msun = munit_cgs/msun_cgs
print(' mass unit (Msun) =', munit_msun)
densunit_msunpc3 = munit_msun/(lunit_pc**3)

### unit conversion
z_pmock = z_pmock/lunit_kpc
z_vmock = z_vmock/lunit_kpc
vz_vmock = vz_vmock/vunit_kms

# plot the input density
# bovy_plot.bovy_print(fig_height=4.,fig_width=6.)
# bovy_plot.bovy_hist(z_pmock,bins=51,normed=True,
#                       xlabel=r'$z$ (pc)',ylabel=r'$\nu(z)$',lw=2.,
#                       histtype='step')
# plt.yscale('log')
# bovy_plot.bovy_end_print('input_density.jpg')

### Set smoothing length and observation bins
h_obs= 0.05
z_obs= numpy.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                    0.55, 0.6, 0.7,
                    -0.1, -0.15, -0.2, -0.25, -0.3, -0.35, -0.4, -0.45,
                    -0.5, -0.55, -0.6, -0.65, -0.7])
### plot range
zmin = -1.1
zmax = 1.1
densmin = 0.01
densmax = 10.0
v2min = 0.0
v2max = 800.0
vmin = -3.0
vmax = 3.0

### observe density
dens_obs= xnm_true*hom2m.compute_dens(z_pmock,zsun_true,z_obs,h_obs,w=m_pmock)
ns_obs= hom2m.compute_nstar(z_pmock,zsun_true,z_obs,h_obs,w=m_pmock)
dens_obs_noise= dens_obs/numpy.sqrt(ns_obs)
print(' dens uncertainties=', dens_obs_noise)

### observe v^2 using MAD of v
v_obs, nv_obs, madv_obs = hom2m.compute_medv(z_vmock,vz_vmock,zsun_true,z_obs,h_obs)
kmad = 1.4826
# use constant noise for v_obs
v_obs_noise= kmad*madv_obs/numpy.sqrt(nv_obs)
# v^2
v2_obs=(kmad*madv_obs)**2
v2_obs_noise = v2_obs/numpy.sqrt(nv_obs)
print(' <v^2>=', v2_obs)
print(' <v^2> noise=',v2_obs_noise)

### set the target data
dens_data= {'type':'dens','pops':0,'zobs':z_obs,'obs':dens_obs,'unc':dens_obs_noise,'zrange':1.}
v2_data= {'type':'v2','pops':0,'zobs':z_obs,'obs':v2_obs,'unc':v2_obs_noise,'zrange':1.}
# v_data= {'type':'v','pops':0,'zobs':z_obs,'obs':v_obs,'unc':v_obs_noise,'zrange':1.}
### only use density and <v^2>
data_dicts= [dens_data,v2_data]
print('zobs=',data_dicts['type'=='dens']['zobs'])

##### M2M fitting both omega and Xnm

### set the initial model
n_m2m= 4000
sigma_init= 20.0
h_m2m= 0.05
# set a guess
xnm_m2m = 0.0025
# total surface_mass density  Munit/Lunit(kpc)^2 
totmass_init = 1500.0 
print(' initial mass density (Msun/pc^2) =', totmass_init*munit_msun/1.0e6)
zh_init = sigma_init**2./totmass_init  # Where 2\pi G= 1 so units of zh are ~311 pc
tdyn = zh_init/sigma_init
print(' initial zh (kpc), tdyn (Myr)=', zh_init, tdyn*yr_s/1.0e6)
z_m2m, vz_m2m, w_init= wendym2m.sample_sech2(sigma_init,totmass_init,n=n_m2m)
z_out= numpy.linspace(zmin, zmax, 101)
dens_init= xnm_m2m*hom2m.compute_dens(z_m2m,zsun_true,z_out,h_m2m,w=w_init)
v2_init= hom2m.compute_v2(z_m2m,vz_m2m,zsun_true,z_out,h_m2m,w=w_init)
v_init= hom2m.compute_v(z_m2m,vz_m2m,zsun_true,z_out,h_m2m,w=w_init)
bovy_plot.bovy_print(axes_labelsize=17.,text_fontsize=12.,xtick_labelsize=15.,ytick_labelsize=15.)

### Set initial DM density, i.e. omega
# dark matter density (Msun pc^-3)
# conventional value
# rhodm_msunpc3= 0.01
rhodm_msunpc3= 0.03
# omega = sqrt(4 pi G rho_dm), 2 pi G=1, 
omega_m2m = numpy.sqrt(2.0*rhodm_msunpc3/densunit_msunpc3)
print(' initial DM density (Msun pc^-3) =', rhodm_msunpc3)
print(' omega(initial)=', omega_m2m)

### Set M2M parameters
step= 0.05*tdyn
nstep= 100
# eps weight, omega, xnm
eps = [10.0**-1.0, 10.0**-0.0, 10.0**-8.0]
smooth= None #1./step/100.
st96smooth= False
mu= 0.
h_m2m= 0.05
zsun_m2m= zsun_true
fit_omega = True
skipomega= 10
skipxnm = 100
fit_xnm = True
prior= 'entropy'
use_v2=True

### Run M2M
w_out,omega_out,xnm_out,z_m2m,vz_m2m,Q,wevol,windx= \
    wendym2m.fit_m2m(w_init,z_m2m,vz_m2m,omega_m2m,zsun_m2m,data_dicts,npop=1,
                     nstep=nstep,step=step,mu=mu,eps=eps,h_m2m=h_m2m,prior=prior,
                     smooth=smooth,st96smooth=st96smooth,output_wevolution=10,
                     fit_omega=fit_omega,skipomega=skipomega,
                     number_density=True, xnm_m2m=xnm_m2m, fit_xnm=fit_xnm, skipxnm=skipxnm)
w_out= w_out[:,0]

### Print Results
print('##### M2M model results #####')
print(' total mass =', numpy.sum(w_out))
print(' mass surface density (Msun/pc^2) =', numpy.sum(w_out)*munit_msun/1.0e6)
print(' omega fit = ',omega_out[-1])
print(' DM density (Msun/pc^-3) =', (omega_out[-1]**2/2.0)*densunit_msunpc3)
print(' Xnm fit= ', xnm_out[-1])
print("Velocity dispersions:",\
      numpy.sqrt(numpy.sum(w_out*(vz_m2m-numpy.sum(w_out*vz_m2m)/numpy.sum(w_out))**2.)/numpy.sum(w_out)))

### Set final values
z_out= numpy.linspace(zmin, zmax, 101)
v2max = 1000.0
dens_final= xnm_out[-1]*hom2m.compute_dens(z_m2m,zsun_true,z_out,h_m2m,w=w_out)
v2_final= hom2m.compute_v2(z_m2m,vz_m2m,zsun_true,z_out,h_m2m,w=w_out)
v_final= hom2m.compute_v(z_m2m,vz_m2m,zsun_true,z_out,h_m2m,w=w_out)

### Save the results in a file
# tempolary set "true" values
omegadm_true = omega_m2m
totmass_true = totmass_init
zh_true = zh_init
sigma_true = sigma_init

savefilename='xnmomega_BB19rhov2obs_colid'+str(cid)+'.sav'
save_pickles(savefilename,w_out,omega_out,xnm_out,z_m2m,vz_m2m,zsun_true,
             vzsun_true,data_dicts,z_pmock,z_vmock,vz_vmock,v_obs,v_obs_noise, \
             w_init,h_m2m,omega_m2m,xnm_m2m,zsun_m2m,\
             dens_init,v2_init,v_init,\
             nstep,step,tdyn,eps,Q,wevol,windx)

### Plot output
bovy_plot.bovy_print(axes_labelsize=19.,text_fontsize=14.,xtick_labelsize=15.,ytick_labelsize=15., fig_height=6.,fig_width=15.)
# density
plt.subplot(2,3,1)
bovy_plot.bovy_plot(z_out,dens_init,'-',semilogy=True,color=init_color,
                   xlabel=r'$\tilde{z}$',ylabel=r'$\nu_{\mathrm{obs}}(\tilde{z})$',
                   xrange=[zmin, zmax],yrange=[densmin,densmax],gcf=True)
bovy_plot.bovy_plot(z_obs,dens_obs,'o',semilogy=True,overplot=True,color=constraint_color)
bovy_plot.bovy_plot(z_out,dens_final,'-',semilogy=True,overplot=True,zorder=0,color=final_color)
plt.errorbar(z_obs,dens_obs,yerr=dens_obs_noise,marker='None',ls='none',color=constraint_color)
plt.yscale('log',nonposy='clip')
# gca().yaxis.set_major_formatter(FuncFormatter(
#                lambda y,pos: (r'${{:.{:1d}f}}$'.format(int(numpy.maximum(-numpy.log10(y),0)))).format(y)))
plt.subplot(2,3,4)
bovy_plot.bovy_plot(z_out,v2_init,'-',color=init_color,
                    xlabel=r'$\tilde{z}$',ylabel=r'$\langle v_z^2\rangle(\tilde{z})$',
                    xrange=[zmin, zmax],yrange=[v2min,v2max],gcf=True)
bovy_plot.bovy_plot(z_obs,v2_obs,'o',overplot=True,color=constraint_color)                    
bovy_plot.bovy_plot(z_out,v2_final,'-',overplot=True,zorder=0,color=final_color)
plt.errorbar(z_obs,v2_obs,yerr=v2_obs_noise,marker='None',ls='none',color=constraint_color)
# gca().yaxis.set_major_formatter(FuncFormatter(
#                lambda y,pos: (r'${{:.{:1d}f}}$'.format(int(numpy.maximum(-numpy.log10(y),0)))).format(y)))
# omega
plt.subplot(2,3,2)
# omega evolution
bovy_plot.bovy_plot(numpy.linspace(0.,1.,nstep)*nstep*step,omega_out,'-',
                    color=sns.color_palette()[0],
                    yrange=[0.,omega_m2m*2.0],
                    semilogx=True,xlabel=r'$\mathrm{orbits}$',ylabel=r'$\omega(t)$',gcf=True)
# axhline(omegadm_true,ls='--',color='0.65',lw=2.,zorder=0)
# gca().xaxis.set_major_formatter(
#    FuncFormatter(
#        lambda y,pos: 
#        (r'${{:.{:1d}f}}$'.format(int(numpy.maximum(-numpy.log10(y),0)))).format(y)))
plt.subplot(2,3,5)
bovy_plot.bovy_plot(z_out,v_init,'-',gcf=True,
                   xlabel=r'$\tilde{z}$',ylabel=r'$\langle v_z\rangle(\tilde{z})$',
                   xrange=[zmin, zmax],yrange=[vmin,vmax])
bovy_plot.bovy_plot(z_obs,v_obs,'o',overplot=True)
bovy_plot.bovy_plot(z_out,v_final,'-',overplot=True,zorder=0,color=final_color)
plt.errorbar(z_obs,v_obs,yerr=v_obs_noise,marker='None',ls='none',color=sns.color_palette()[1])
# print("Velocity dispersions: mock, fit",numpy.std(vz_mock),\
#      numpy.sqrt(numpy.sum(w_out*(vz_m2m-numpy.sum(w_out*vz_m2m)/numpy.sum(w_out))**2.)/numpy.sum(w_out)))
plt.subplot(2,3,3)
bovy_plot.bovy_plot(numpy.linspace(0.,1.,nstep)*nstep*step,xnm_out,'-',
                    color=sns.color_palette()[0],
                    yrange=[0.,xnm_m2m*2.0],
                    semilogx=True,xlabel=r'$\mathrm{orbits}$',ylabel=r'$X_{nm}(t)$',gcf=True)
# axhline(xnm_true,ls='--',color='0.65',lw=2.,zorder=0)
#gca().xaxis.set_major_formatter(FuncFormatter(
#                lambda y,pos: (r'${{:.{:1d}f}}$'.format(int(numpy.maximum(-numpy.log10(y),0)))).format(y)))
# for ii in range(len(wevol)):
#    bovy_plot.bovy_plot(numpy.linspace(0.,1.,nstep)*nstep*step,wevol[ii,0],'-',
#                        color=cm.viridis(z_m2m[windx][ii]/0.3),
#                        yrange=[-0.2/len(z_m2m),numpy.amax(wevol)*1.1],
#                        semilogx=True,xlabel=r'$t$',ylabel=r'$w(t)$',gcf=True,overplot=ii>0)
#gca().xaxis.set_major_formatter(FuncFormatter(
#                lambda y,pos: (r'${{:.{:1d}f}}$'.format(int(numpy.maximum(-numpy.log10(y),0)))).format(y)))
plt.subplot(2,3,6)
bovy_plot.bovy_plot(numpy.linspace(0.,1.,nstep)*nstep*step,numpy.sum(Q,axis=1),lw=3.,
                   loglog=True,xlabel=r'$t$',ylabel=r'$\chi^2$',gcf=True,
                   yrange=[1.,10**7.0])
#gca().yaxis.set_major_formatter(FuncFormatter(
#                lambda y,pos: (r'${{:.{:1d}f}}$'.format(int(numpy.maximum(-numpy.log10(y),0)))).format(y)))
#gca().xaxis.set_major_formatter(FuncFormatter(
#                lambda y,pos: (r'${{:.{:1d}f}}$'.format(int(numpy.maximum(-numpy.log10(y),0)))).format(y)))
plt.tight_layout()
bovy_plot.bovy_end_print('m2m_results.jpg')



