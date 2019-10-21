
#
# WendyM2M for a perturbed disk mock data
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

### Options
Add_perturbation = True

print(' Add Perturbation = ', Add_perturbation)

### plot range
zmin = -1.1
zmax = 1.1
densmin = 0.1
densmax = 10.0
v2min = 0.0
v2max = 800.0
vmin = -3.0
vmax = 3.0

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
print(' time unit (Myr) = ',tunit_cgs/yr_s/1.0e6)
# mass unit 
munit_cgs = (lunit_cgs/twopigunit_cgs)*(lunit_cgs/tunit_cgs) \
        *(lunit_cgs/tunit_cgs)
munit_msun = munit_cgs/msun_cgs
print(' mass unit (Msun) =', munit_msun)
densunit_msunpc3 = munit_msun/(lunit_pc**3)

### computation parameters
dttdyn = 0.01    # time step is set with dt = dttdyn*tdyn
print(' time step set ', dttdyn,' x tdyn')

##### generate a mock data
##### mock data parameters
n_init = 100000
sigma_true = 17.5 # Velocity dispersion in the disc km/s
totmass_true = 1200.0 # Surface density of the disc (1200 ~ 44 Msun pc^-2)
omegadm_true = 30.4 # external potential (30.4 ~ 0.017 Msun/pc^3)
zsun_true= 0.0  #  The position of the Sun from the disk mid-plane (kpc)
vzsun_true= 0.0  # The Sun' vertical motion. 
xnm_true = 0.002  #  Xnm
h_def = 0.1   # default h
print(' disk parameters, np, sigma, Mtot, omega, zsun, Xnm=', \
      n_init, sigma_true, totmass_true, omegadm_true, zsun_true, xnm_true)
print(' smoothing length (default) =', h_def)

n_init0 = n_init
zh_true = sigma_true**2./totmass_true  # Where 2\pi G= 1 so units of zh are ~311 pc
tdyn = zh_true/sigma_true
z_init, vz_init, m_init = wendym2m.sample_sech2(
    sigma_true, totmass_true, n=n_init)
print('zh, tdyn, omega_dm =', zh_true, tdyn, omegadm_true)

# run Wendy to introduce the background potential adiabatically
print(' running Wendy to add the background potential adiabatically')
g = wendypy.nbody(z_init, vz_init, m_init, dttdyn*tdyn, approx=True, nleap=1)
nt = 700
zt = numpy.empty((n_init, nt+1))
vzt = numpy.empty((n_init, nt+1))
# Et = numpy.empty((nt+1))
zt[:, 0] = z_init
vzt[:, 0] = vz_init
# Et[0] = wendy.energy(z_init, vz_init, m_init)
# increasing omega
nstep_omega = 500
domega = omegadm_true/nstep_omega
omega_ii = 0.0
tz = z_init
tvz = vz_init
for ii in range(nt):
    if ii <= nstep_omega:
        g = wendypy.nbody(tz, tvz, m_init, dttdyn*tdyn, omega=omega_ii, approx=True, nleap=1)
    tz, tvz= next(g)
    zt[:, ii+1] = tz
    vzt[:, ii+1] = tvz
    # Et[ii+1] = wendy.energy(tz, tvz, m_init, omega=omega_ii)
    # update omega
    if ii < nstep_omega:
        omega_ii += domega
z_start= zt[:, -1]
vz_start= vzt[:, -1]
# print('Final omega and omega_dm =', omega_ii, omegadm_true)

# Perturb a the disk by adding a perturber passing through the disk
if Add_perturbation==True:
  print(' Running Wendy with a perturber to perturbe the disk')
  nt = 2000  # number of timestep
  vz0sat = 20 # The speed of the satellite unit is km/s
  pmdisk=0.15  # The mass of the satellite as a percentage of disc mass
  msat = totmass_true*pmdisk
  z0sat = -3.0 # The initial location of the satellite from the disk plane (kpc)
  print(' Perturber initial parameters, vz, m and zini=', vz0sat, msat, z0sat)
  dt = dttdyn*tdyn
  i_remove = int(round((max(z_start)*1.1-z0sat)/vz0sat/dt))

  z = copy.deepcopy(z_start)
  vz = copy.deepcopy(vz_start)

  # adding the satellite
  n_init = n_init0+1
  ntot_sat = n_init
  z = numpy.append(z, z0sat)
  vz = numpy.append(vz, vz0sat)
  m = numpy.append(m_init, msat)

  print('number of times step and step of removal of the perturber =', \
        nt, i_remove)

  zt = numpy.empty((n_init, nt+1))
  vzt = numpy.empty((n_init, nt+1))
  # Et = numpy.empty((n_init))
  zt[:,0] = z
  vzt[:,0] = vz
  # Et[0] = wendy.energy(z, vz, m, omega=omegadm_true)

  g= wendypy.nbody(z, vz, m, dt, omega=omegadm_true, approx=True, nleap=1)

  for ii in range(i_remove):
    tz, tvz = next(g)
    tz[-1] = z0sat+ii*dt*vz0sat
    tvz[-1] = vz0sat
    zt[:,ii+1] = tz
    vzt[:,ii+1] = tvz
    # Et[ii+1] = wendy.energy(tz, tvz, m)
    g = wendypy.nbody(tz, tvz, m, dt, omega=omegadm_true, approx=True, nleap=1)
    
  g = wendypy.nbody(tz[:-1], tvz[:-1], m[:-1], dt, omega=omegadm_true, approx=True, nleap=5)
  for ii in range(nt-i_remove):
    tz, tvz = next(g)
    zt[:-1, ii+i_remove+1] = tz
    vzt[:-1, ii+i_remove+1] = tvz
    # Et[ii+i_remove+1] = wendy.energy(tz, tvz, m[:-1], omega=omegadm_true, twopiG=2.*pi)

  # adjust the position
  zt_iso1 = numpy.array([zt[:, i]-numpy.median(zt[:, i]) for i in range(i_remove)])
  zt_iso2 = numpy.array([zt[:-1, i+i_remove]-numpy.median(zt[:-1, i+i_remove]) 
                 for i in range(nt-i_remove+1)])

  zt_iso = numpy.zeros((ntot_sat, nt+1))
  zt_iso[:, :i_remove] = zt_iso1.T
  zt_iso[:-1, i_remove:] = zt_iso2.T
  zt_iso[-1, i_remove:] = numpy.nan
  n_init = n_init0

### Choose the target data snapshot ###
n_mock = n_init
if Add_perturbation==True:
  istep = 1800
  print('target snapshot is chosen at step=', istep)
  z_mock = zt_iso[:-1, istep]
  vz_mock = vzt[:-1, istep]-numpy.mean(vzt[:-1, istep])
else:
  print('targe mock data from a stablized disk after adding a DM background.')
  z_mock = z_start
  vz_mock = vz_start

m_mock = m_init
totmass_true = numpy.sum(m_mock)
omegadm = copy.deepcopy(omegadm_true)

# save the mock snapshot data
if Add_perturbation==True:
  savefilename='xnmomega_perturbed_target_mzvz.sav'
else:
  savefilename='xnmomega_stable_target_mzvz.sav'  
save_pickles(savefilename, m_mock, z_mock, vz_mock, omegadm_true)

# The z positions of observation
z_obs= numpy.array([0.15, 0.25, 0.35, 0.45, 0.55, 0.65,
                    -0.15, -0.25, -0.35, -0.45, -0.55, -0.65])
h_obs= h_def
# density
dens_obs= xnm_true*hom2m.compute_dens(z_mock,zsun_true,z_obs,h_obs,w=m_mock)
dens_obs_noise= 0.2*dens_obs/numpy.sqrt(dens_obs)
numpy.random.seed(203)
dens_obs+= numpy.random.normal(size=dens_obs.shape)*dens_obs_noise
# v^2
v2_obs= hom2m.compute_v2(z_mock,vz_mock,zsun_true,z_obs,h_obs)
# use constant noise
v2_obs_noise=numpy.zeros_like(v2_obs)+20.0
numpy.random.seed(10) # probably best to set a seed somewhere so the data is always the same
v2_obs+= numpy.random.normal(size=v2_obs.shape)*v2_obs_noise

# <v>, but not used for constraints.
# We only observe the v2 at a few z (same as before)
v_obs= hom2m.compute_v(z_mock,vz_mock,zsun_true,z_obs,h_obs)
# use constant noise
v_obs_noise=numpy.zeros_like(v_obs)+0.5
numpy.random.seed(42) # probably best to set a seed somewhere so the data is always the same
v_obs+= numpy.random.normal(size=v_obs.shape)*v_obs_noise

### setting data_dicts as the target data for M2M modelling
dens_data= {'type':'dens','pops':0,'zobs':z_obs,'obs':dens_obs,'unc':dens_obs_noise,'zrange':1.}
v2_data= {'type':'v2','pops':0,'zobs':z_obs,'obs':v2_obs,'unc':v2_obs_noise,'zrange':1.}
v_data= {'type':'v','pops':0,'zobs':z_obs,'obs':v_obs,'unc':v_obs_noise,'zrange':1.}
data_dicts= [dens_data,v2_data]
print('zobs=',data_dicts['type'=='dens']['zobs'])

##### M2M fitting both omega and Xnm

### set the initial model
n_m2m= 4000
print(' Number of M2M model particles =', n_m2m)
sigma_init= sigma_true*1.2
h_m2m= h_def
# set a guess
xnm_m2m = xnm_true*1.5
# total surface_mass density  Munit/Lunit(kpc)^2
totmass_init = totmass_true*1.5
print(' initial mass density (Msun/pc^2) =', \
      totmass_init*munit_msun/(lunit_pc**2))
zh_init = sigma_init**2./totmass_init  
tdyn = zh_init/sigma_init
print(' initial zh, tdyn =', zh_init, tdyn)
# DM density
omega_m2m = omegadm_true*1.5
print('Initial omega, Xnm and sigma =', omega_m2m, xnm_m2m, sigma_init)

z_m2m, vz_m2m, w_init= wendym2m.sample_sech2(sigma_init,totmass_init,n=n_m2m)
z_out= numpy.linspace(zmin, zmax, 101)
dens_init= xnm_m2m*hom2m.compute_dens(z_m2m,zsun_true,z_out,h_m2m,w=w_init)
v2_init= hom2m.compute_v2(z_m2m,vz_m2m,zsun_true,z_out,h_m2m,w=w_init)
v_init= hom2m.compute_v(z_m2m,vz_m2m,zsun_true,z_out,h_m2m,w=w_init)
bovy_plot.bovy_print(axes_labelsize=17.,text_fontsize=12.,xtick_labelsize=15.,ytick_labelsize=15.)

### adiabatically add omega to the initial model
g = wendypy.nbody(z_m2m, vz_m2m, w_init, dttdyn*tdyn, approx=True, nleap=1)
nt = 600
print('Addiabatically adding omega, nt=', nt)
zt = numpy.empty((n_m2m, nt+1))
vzt = numpy.empty((n_m2m, nt+1))
# Et = numpy.empty((nt+1))
zt[:, 0] = z_m2m
vzt[:, 0] = vz_m2m
# Et[0] = wendy.energy(z_init, vz_init, m_init)
# increasing omega
nstep_omega = 500
domega = omega_m2m/nstep_omega
omega_ii = 0.0
tz = z_m2m
tvz = vz_m2m
for ii in range(nt):
    if ii <= nstep_omega:
        g = wendypy.nbody(tz, tvz, w_init, dttdyn*tdyn, omega=omega_ii, approx=True, nleap=1)
    tz, tvz= next(g)
    zt[:, ii+1] = tz
    vzt[:, ii+1] = tvz
    # Et[ii+1] = wendy.energy(tz, tvz, m_init, omega=omega_ii)
    # update omega
    if ii < nstep_omega:
        omega_ii += domega
print('Final omega and omega_dm =', omega_ii, omegadm_true)

### Set M2M parameters
step= dttdyn*tdyn
nstep= 10000
# eps weight, omega, xnm
eps = [10.0**1.0, 10.0**2.5, 10.0**-9.0]
print('M2M parameters: nstep, eps =', nstep, eps)
smooth= None #1./step/100.
st96smooth= False
mu= 0.
h_m2m= h_def
zsun_m2m= zsun_true
fit_omega = True
skipomega= 10
skipxnm = 100
fit_xnm = True
prior= 'entropy'
use_v2=True
print('skipomega,skipxnm =', skipomega, skipxnm)
print(' ft omega, xnm =', fit_omega, fit_xnm)
print(' smooth, st96smooth, prior, use_v2=',smooth, st96smooth, prior, use_v2)
print(' mu, h_m2m=', mu, h_m2m)

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
print(' total mass fit, true=', numpy.sum(w_out), totmass_true)
print(' omega fit, true = ',omega_out[-1], omegadm_true)
print(' Xnm fit, true= ', xnm_out[-1], xnm_true)
print("Velocity dispersions:",\
      numpy.sqrt(numpy.sum(w_out*(vz_m2m-numpy.sum(w_out*vz_m2m)/numpy.sum(w_out))**2.)/numpy.sum(w_out)))

### Set final values
z_out= numpy.linspace(zmin, zmax, 101)
dens_final= xnm_out[-1]*hom2m.compute_dens(z_m2m,zsun_true,z_out,h_m2m,w=w_out)
v2_final= hom2m.compute_v2(z_m2m,vz_m2m,zsun_true,z_out,h_m2m,w=w_out)
v_final= hom2m.compute_v(z_m2m,vz_m2m,zsun_true,z_out,h_m2m,w=w_out)

### Plot output
bovy_plot.bovy_print(axes_labelsize=19.,text_fontsize=14.,xtick_labelsize=15.,ytick_labelsize=15., fig_height=6.,fig_width=15.)
# density
plt.subplot(2,3,1)
bovy_plot.bovy_plot(z_out,dens_init,'--',semilogy=True,color=init_color,
                   xlabel=r'$\tilde{z}$',ylabel=r'$\nu_{\mathrm{obs}}(\tilde{z})$',
                   xrange=[zmin, zmax],yrange=[densmin,densmax],gcf=True)
bovy_plot.bovy_plot(z_obs,dens_obs,'o',semilogy=True,overplot=True,color=constraint_color)
bovy_plot.bovy_plot(z_out,dens_final,'-',semilogy=True,overplot=True,zorder=0,color=final_color)
plt.errorbar(z_obs,dens_obs,yerr=dens_obs_noise,marker='None',ls='none',color=constraint_color)
plt.yscale('log',nonposy='clip')
# gca().yaxis.set_major_formatter(FuncFormatter(
#                lambda y,pos: (r'${{:.{:1d}f}}$'.format(int(numpy.maximum(-numpy.log10(y),0)))).format(y)))

# v^2
plt.subplot(2,3,4)
bovy_plot.bovy_plot(z_out,v2_init,'--',color=init_color,
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
plt.axhline(omegadm_true,ls='--',color='0.65',lw=2.,zorder=0)
plt.gca().xaxis.set_major_formatter(
    FuncFormatter(
        lambda y,pos: 
        (r'${{:.{:1d}f}}$'.format(int(numpy.maximum(-numpy.log10(y),0)))).format(y)))

# <v>
plt.subplot(2,3,5)
bovy_plot.bovy_plot(z_out,v_init,'--',gcf=True,
                   xlabel=r'$\tilde{z}$',ylabel=r'$\langle v_z\rangle(\tilde{z})$',
                   xrange=[zmin, zmax],yrange=[vmin,vmax])
bovy_plot.bovy_plot(z_obs,v_obs,'o',overplot=True)
bovy_plot.bovy_plot(z_out,v_final,'-',overplot=True,zorder=0,color=final_color)
plt.errorbar(z_obs,v_obs,yerr=v_obs_noise,marker='None',ls='none',color=sns.color_palette()[1])
# print("Velocity dispersions: mock, fit",numpy.std(vz_mock),\
#      numpy.sqrt(numpy.sum(w_out*(vz_m2m-numpy.sum(w_out*vz_m2m)/numpy.sum(w_out))**2.)/numpy.sum(w_out)))

# Xnm
plt.subplot(2,3,3)
bovy_plot.bovy_plot(numpy.linspace(0.,1.,nstep)*nstep*step,xnm_out,'-',
                    color=sns.color_palette()[0],
                    yrange=[0.,xnm_m2m*2.0],
                    semilogx=True,xlabel=r'$\mathrm{orbits}$',ylabel=r'$X_{nm}(t)$',gcf=True)
plt.axhline(xnm_true,ls='--',color='0.65',lw=2.,zorder=0)
plt.gca().xaxis.set_major_formatter(FuncFormatter(
                lambda y,pos: (r'${{:.{:1d}f}}$'.format(int(numpy.maximum(-numpy.log10(y),0)))).format(y)))
# for ii in range(len(wevol)):
#    bovy_plot.bovy_plot(numpy.linspace(0.,1.,nstep)*nstep*step,wevol[ii,0],'-',
#                        color=cm.viridis(z_m2m[windx][ii]/0.3),
#                        yrange=[-0.2/len(z_m2m),numpy.amax(wevol)*1.1],
#                        semilogx=True,xlabel=r'$t$',ylabel=r'$w(t)$',gcf=True,overplot=ii>0)
#gca().xaxis.set_major_formatter(FuncFormatter(
#                lambda y,pos: (r'${{:.{:1d}f}}$'.format(int(numpy.maximum(-numpy.log10(y),0)))).format(y)))

# Xi^2
plt.subplot(2,3,6)
bovy_plot.bovy_plot(numpy.linspace(0.,1.,nstep)*nstep*step,numpy.sum(Q,axis=1),lw=3.,
                   loglog=True,xlabel=r'$t$',ylabel=r'$\chi^2$',gcf=True,
                   yrange=[1.,10**7.0])
#gca().yaxis.set_major_formatter(FuncFormatter(
#                lambda y,pos: (r'${{:.{:1d}f}}$'.format(int(numpy.maximum(-numpy.log10(y),0)))).format(y)))
#gca().xaxis.set_major_formatter(FuncFormatter(
#                lambda y,pos: (r'${{:.{:1d}f}}$'.format(int(numpy.maximum(-numpy.log10(y),0)))).format(y)))
plt.tight_layout()
if Add_perturbation==True:
  bovy_plot.bovy_end_print('m2m_results_perturbed_mock.jpg')
else:
  bovy_plot.bovy_end_print('m2m_results_stable_mock.jpg')  

### Save the results in a file
if Add_perturbation==True:
  savefilename='m2m_results_perturbed_mock.sav'
else:
  savefilename='m2m_results_stable_mock.sav'  
save_pickles(savefilename,w_out,omega_out,xnm_out,z_m2m,vz_m2m,zsun_true,
             vzsun_true,data_dicts,z_mock,vz_mock,v_obs,v_obs_noise, \
             w_init,h_m2m,omega_m2m,xnm_m2m,zsun_m2m,\
             dens_init,v2_init,v_init,\
             h_obs,xnm_true,omegadm_true,totmass_true,zh_true,sigma_true,
             nstep,step,tdyn,skipomega,skipxnm,dttdyn,eps,Q,wevol,windx)
