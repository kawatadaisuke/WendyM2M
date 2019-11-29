
#
# Sample WendyM2M for mock data and BB19 Gaia DR2 data
#
# sample both omega and Xnm
#

import os
import sys
import numpy
from scipy.misc import logsumexp
import tqdm
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
import matplotlib.animation as animation
from matplotlib import gridspec
from IPython.display import HTML, Image
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

### set the model results
# obs_data = 'mock_stable'
obs_data = 'mock_perturbed'
# obs_data = 'BB19_GDR2'

print(' target observational data =', obs_data)

### Colour ID
cid = 0

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
yr_s = ((365.0*24.0*3600.0)*3+(366.0*24.0*3600.0))/4.0
# mass unit 
munit_cgs = lunit_cgs**3/(twopigunit_cgs*(tunit_cgs**2))
munit_msun = munit_cgs/msun_cgs
densunit_msunpc3 = munit_msun/(lunit_pc**3)

### computation parameters
dttdyn = 0.01    # time step is set with dt = dttdyn*tdyn
print(' time step set ', dttdyn,' x tdyn')
h_def = 0.1
print(' default h =', h_def)

### Read M2M model results
if obs_data == 'mock_stable':
  savefilename= 'm2m_results_stable_mock.sav'
elif obs_data == 'mock_perturbed':
  savefilename= 'm2m_results_perturbed_mock.sav'
else:
  savefilename= 'xnmomega_BB19rhov2obs_colid'+str(cid)+'.sav'
print(' reading M2M fit results ',savefilename)

if os.path.exists(savefilename):
    with open(savefilename,'rb') as savefile:
        out= (pickle.load(savefile),)
        while True:
            try:
                out= out+(pickle.load(savefile),)
            except EOFError:
                break
    if obs_data=='mock_stable' or obs_data=='mock_perturbed':
      w_out,omega_out,xnm_out,z_m2m,vz_m2m,zsun_true, \
             vzsun_true,data_dicts,z_mock,vz_mock,v_obs,v_obs_noise, \
             w_init,h_m2m,omega_m2m,xnm_m2m,zsun_m2m,\
             dens_init,v2_init,v_init,\
             h_obs,xnm_true,omegadm_true,totmass_true,zh_true,sigma_true, \
             nstep,step,tdyn,skipomega,skipxnm,dttdyn,eps,Q,wevol,windx = out
      vz_vmock = vz_mock
    else:
      w_out,omega_out,xnm_out,z_m2m,vz_m2m,zsun_true,vzsun_true, \
      data_dicts,z_pmock,z_vmock,vz_vmock,v_obs,v_obs_noise, \
      w_init,h_m2m,omega_m2m,xnm_m2m,zsun_m2m,dens_init,v2_init,v_init, \
      nstep,step,tdyn,eps,Q,wevol,windx = out

if h_m2m!=h_def:
  print('Error h_m used =', h_m2m,' but h_def =', h_def)
  sys.exit()

for jj,data_dict in enumerate(data_dicts):
    if data_dict['type'].lower() == 'dens':
        z_obs = data_dict['zobs']
        dens_obs = data_dict['obs']
        dens_obs_noise = data_dict['unc']
    elif data_dict['type'].lower() == 'v2':
        v2_obs = data_dict['obs']
        v2_obs_noise = data_dict['unc']
        
### Test Output M2M results
print('##### M2M model results #####')
if obs_data=='mock_stable' or obs_data=='mock_perturbed':
  print(' total mass fit, true=', numpy.sum(w_out), totmass_true)
  print(' omega fit, true = ',omega_out[-1], omegadm_true)
  print(' Xnm fit, true= ', xnm_out[-1], xnm_true)
  
print(' total mass =', numpy.sum(w_out))
print(' mass surface density (Msun/pc^2) =', numpy.sum(w_out)*munit_msun/1.0e6)
print(' omega fit = ',omega_out[-1])
print(' DM density (Msun/pc^-3) =', (omega_out[-1]**2/2.0)*densunit_msunpc3)
print(' Xnm fit= ', xnm_out[-1])
print("Velocity dispersions:",\
      numpy.sqrt(numpy.sum(w_out*(vz_m2m-numpy.sum(w_out*vz_m2m)/numpy.sum(w_out))**2.)/numpy.sum(w_out)))

### Set plot range
zmin = -1.1
zmax = 1.1
densmin = 0.1
densmax = 10.0
v2min = 0.0
v2max = 800.0
# for <v>
vmin = -6.0
vmax = 6.0
# for vz histogram
vzmin = -50.0
vzmax = 50.0
sfmdenmin = 0.0
sfmdenmax = 50.0
dmdenmin = 0.0
dmdenmax = 0.03

### Set final values
z_out= numpy.linspace(zmin, zmax, 101)
v2max = 1000.0
dens_final= xnm_out[-1]*hom2m.compute_dens(z_m2m,zsun_true,z_out,h_m2m,w=w_out)
v2_final= hom2m.compute_v2(z_m2m,vz_m2m,zsun_true,z_out,h_m2m,w=w_out)
v_final= hom2m.compute_v(z_m2m,vz_m2m,zsun_true,z_out,h_m2m,w=w_out)

### Sample w, omega and Xnm
# default sample step
step_sam= dttdyn*tdyn
nstep_sam = 500
eps = [10.0**-1.0, 10.0**2.5, 10.0**-10.5]
print('M2M sample parameters: nstep_sam, eps =', nstep_sam, eps)
# used for weight sampling only
eps_sam = eps[0]
nsamples= 100 
s_low, s_high= 16, -16
print(' Nsample =', nsamples)
smooth= None #1./step/100.
st96smooth= False
mu= 0.0
h_m2m= h_def
omega_m2m= omega_out[-1]
zsun_m2m= zsun_true
xnm_m2m = xnm_out[-1]
fit_zsun = False
fit_omega = True
# not fraction, but 1 sigma size for step of MCMC
# note omega's scale is 40 or so
sig_omega = 30.4*1.0e-3
nmh_omega = 25
fit_xnm = True
# not fraction, but 1 sigma size for step of MCMC
# note Xnm scale is aroud 0.002
sig_xnm = 0.002*1.0e-3
nmh_xnm = 25
prior= 'entropy'
use_v2=True
# these will not be used
skipomega= 10
skipxnm = 100

print('skipomega,skipxnm =', skipomega, skipxnm)
print(' ft omega, xnm =', fit_omega, fit_xnm)
print(' sig_omega, nmh_omega =', sig_omega, nmh_omega)
print(' sig_xnm, nmh_omega =', sig_xnm, nmh_xnm)
print(' smooth, st96smooth, prior, use_v2=',smooth, st96smooth, prior, use_v2)
print(' mu, h_m2m=', mu, h_m2m)

if obs_data=='mock_stable':
    savefilename= 'sam_stable_mock.sav'  
elif obs_data=='mock_perturbed':  
    savefilename= 'sam_perturbed_mock.sav'
else:
    savefilename= 'sam_wxnmomega_BB19rhov2obs_colid'+str(cid)+'.sav'
print(' check if there is sample results file ',savefilename)

if os.path.exists(savefilename):
    with open(savefilename,'rb') as savefile:
        out= (pickle.load(savefile),)
        while True:
            try:
                out= out+(pickle.load(savefile),)
            except EOFError:
                break
else:
    out= wendym2m.sample_m2m(nsamples,w_out,z_m2m,vz_m2m,omega_m2m,zsun_m2m,
                         data_dicts,
                         nstep=nstep_sam,step=step_sam,eps=eps_sam,
                         mu=mu,h_m2m=h_m2m,prior=prior,w_prior=w_init,
                         smooth=smooth,st96smooth=st96smooth,
                         fit_omega=fit_omega, sig_omega=sig_omega,
                         nmh_omega=nmh_omega, skipomega=skipomega,
                         number_density=True, xnm_m2m=xnm_m2m,
                         fit_xnm=fit_xnm, skipxnm=skipxnm,
                         sig_xnm=sig_xnm, nmh_xnm=nmh_xnm, fix_weights=False)
    save_pickles(savefilename,*out)
w_sam,xnm_sam, omega_sam, Q_sam,z_sam,vz_sam= out

### Output the results
print("#####   Results after sampling   #####")
# for test
s_low=-8
s_high=8
#
xnm_m2m = xnm_out[-1]
omega_mean = numpy.mean(omega_sam)
omega_std = numpy.std(omega_sam)
dmden_sam = (omega_sam**2/2.0)*densunit_msunpc3
omega_m2m = omega_out[-1]
# pick up first population
w_samallpop = copy.deepcopy(w_sam)
w_sam = copy.deepcopy(w_samallpop[:, :, 0])
# print(' shape z, vz, w=', numpy.shape(z_sam), numpy.shape(vz_sam), numpy.shape(w_sam))
totmass_sam = numpy.sum(w_sam[:,:], axis=1)
# surface mass density of stars
sfmden_star_sam = totmass_sam*munit_msun/(lunit_pc**2)

if obs_data=='mock_stable' or obs_data=='mock_perturbed':
  print('xnm: true, initial, best-fit, mean of samples, unc.=',xnm_true, \
        xnm_out[-1],numpy.mean(xnm_sam),numpy.std(xnm_sam))
  print('omega: true, best-fit, initial, mean of samples, unc.=',omegadm_true, \
        omega_out[-1],omega_mean,omega_std)
  dmden_true = (omegadm_true**2/2.0)*densunit_msunpc3
  print(' DM density (Msun/pc^-3), true, mean  +- unc =', dmden_true, \
      (omega_mean**2/2.0)*densunit_msunpc3, \
      (((omega_mean+omega_std)**2-omega_mean**2)/2.0)*densunit_msunpc3, \
      ((omega_mean**2-(omega_mean-omega_std)**2)/2.0)*densunit_msunpc3)
  sfmden_true = totmass_true*munit_msun/(lunit_pc**2)
  print(' Stellar mass surface density (Msun/pc^2) true, mean, unc =', \
        sfmden_true,
        numpy.mean(sfmden_star_sam),numpy.std(sfmden_star_sam))
else: 
  print('xnm: best-fit, mean of samples unc.)',xnm_out[-1],numpy.mean(xnm_sam),numpy.std(xnm_sam))
  print('omega: best-fit, mean of samples unc.)',omega_out[-1],omega_mean, \
      omega_std)
  print(' DM density (Msun/pc^-3), mean and +- unc =', \
      (omega_mean**2/2.0)*densunit_msunpc3, \
      (((omega_mean+omega_std)**2-omega_mean**2)/2.0)*densunit_msunpc3, \
      ((omega_mean**2-(omega_mean-omega_std)**2)/2.0)*densunit_msunpc3)
  print(' Stellar mass surface density (Msun/pc^2) mean, unc =', \
      numpy.mean(sfmden_star_sam),numpy.std(sfmden_star_sam))
# print(' size of sample stellar mass and DM density=', \
#       numpy.shape(sfmden_star_sam), numpy.shape(dmden_sam))

# final z profiles
z_out= numpy.linspace(zmin, zmax, 101)
dens_final= xnm_m2m*hom2m.compute_dens(z_m2m,zsun_true,z_out,h_m2m,w=w_out)
v2_final= hom2m.compute_v2(z_m2m,vz_m2m,zsun_true,z_out,h_m2m,w=w_out)
v_final= hom2m.compute_v(z_m2m,vz_m2m,zsun_true,z_out,h_m2m,w=w_out)
# density and v2 for samples
dens_final_sam= numpy.empty((nsamples,len(dens_final)))
v2_final_sam= numpy.empty((nsamples,len(dens_final)))
v_final_sam= numpy.empty((nsamples,len(dens_final)))
vz_hist = numpy.empty((nsamples,31))
for ii in range(nsamples):
    dens_final_sam[ii]= xnm_sam[ii]*hom2m.compute_dens(z_sam[ii],zsun_true,z_out,h_m2m,w=w_sam[ii])
    v2_final_sam[ii]= hom2m.compute_v2(z_sam[ii],vz_sam[ii],zsun_true,z_out,h_m2m,w=w_sam[ii])
    v_final_sam[ii]= hom2m.compute_v(z_sam[ii],vz_sam[ii],zsun_true,z_out,h_m2m,w=w_sam[ii])
    vz_hist[ii], _= numpy.histogram(vz_sam[ii],weights=w_sam[ii], \
                                    density=True,bins=31,range=[vzmin,vzmax]) 
dens_final_sam_sorted= numpy.sort(dens_final_sam,axis=0)
v2_final_sam_sorted= numpy.sort(v2_final_sam,axis=0)
v_final_sam_sorted= numpy.sort(v_final_sam,axis=0)
w_sam_sorted= numpy.sort(w_sam,axis=0)
vz_hist_sorted= numpy.sort(vz_hist,axis=0)
# |z| vs. rho_total
zabs_out=numpy.linspace(0.0, zmax, 50)
if obs_data=='mock_stable' or obs_data=='mock_perturbed':
  # set true mass profile
  sfmden_z_tot_true = numpy.zeros_like(zabs_out)
  sfmden_z_star_true = numpy.zeros_like(zabs_out)
  sfmden_z_dm_true = numpy.zeros_like(zabs_out)
  # weight for mock
  m_mock = totmass_true/len(z_mock)
  print(' mock data weight =',m_mock)
  for jj,zlim in enumerate(zabs_out):
    indx = numpy.where(z_mock<zlim)
    sfmden_z_star_true[jj] = m_mock*len(z_mock[indx])*munit_msun/(lunit_pc**2)
  sfmden_z_dm_true = zabs_out*(omegadm_true**2/2.0) \
    *densunit_msunpc3*lunit_pc
  sfmden_z_tot_strue = sfmden_z_star_true+sfmden_z_dm_true

sfmden_z_tot_sam = numpy.zeros((nsamples,len(zabs_out)))
sfmden_z_star_sam = numpy.zeros((nsamples,len(zabs_out)))
sfmden_z_dm_sam = numpy.zeros((nsamples,len(zabs_out)))
zabs_sam = numpy.abs(z_sam[:, :])
for ii in range(nsamples):
  for jj,zlim in enumerate(zabs_out):
    indx = numpy.where(zabs_sam[ii, :]<zlim)
    w_selp = w_sam[ii, indx]
    sfmden_z_star_sam[ii, jj] = numpy.sum(w_selp)*munit_msun/(lunit_pc**2)
  sfmden_z_dm_sam[ii,:] = zabs_out*(omega_sam[ii]**2/2.0) \
    *densunit_msunpc3*lunit_pc
  sfmden_z_tot_sam[ii,:] = sfmden_z_star_sam[ii,:]+sfmden_z_dm_sam[ii,:]

sfmden_z_star_sam_sorted= numpy.sort(sfmden_z_star_sam,axis=0)
sfmden_z_dm_sam_sorted= numpy.sort(sfmden_z_dm_sam,axis=0)
sfmden_z_tot_sam_sorted= numpy.sort(sfmden_z_tot_sam,axis=0)

### plot
bovy_plot.bovy_print(axes_labelsize=19.,text_fontsize=14., \
                     xtick_labelsize=15.,ytick_labelsize=15., \
                     fig_height=10.,fig_width=15.)
# density
plt.subplot(2,3,1)
bovy_plot.bovy_plot(z_out,dens_init,'-',semilogy=True,color=init_color,
                   xlabel=r'$\tilde{z}$',ylabel=r'$\nu_{\mathrm{obs}}(\tilde{z})$',
                   xrange=[zmin, zmax],yrange=[densmin,densmax],gcf=True)
bovy_plot.bovy_plot(z_obs,dens_obs,'o',semilogy=True,overplot=True,color=constraint_color)
# bovy_plot.bovy_plot(z_out,dens_final,'-',semilogy=True,overplot=True,zorder=0,color=final_color)
plt.errorbar(z_obs,dens_obs,yerr=dens_obs_noise,marker='None',ls='none',color=constraint_color)
plt.fill_between(z_out,dens_final_sam_sorted[s_low],dens_final_sam_sorted[s_high],color='0.65',zorder=0)
plt.yscale('log',nonposy='clip')

### <v^2>
plt.subplot(2,3,4)
bovy_plot.bovy_plot(z_out,v2_init,'-',color=init_color,
                    xlabel=r'$\tilde{z}$',ylabel=r'$\langle v_z^2\rangle(\tilde{z})$',
                    xrange=[zmin, zmax],yrange=[v2min,v2max],gcf=True)
bovy_plot.bovy_plot(z_obs,v2_obs,'o',overplot=True,color=constraint_color)                    
# bovy_plot.bovy_plot(z_out,v2_final,'-',overplot=True,zorder=0,color=final_color)
plt.errorbar(z_obs,v2_obs,yerr=v2_obs_noise,marker='None',ls='none',color=constraint_color)
plt.fill_between(z_out, v2_final_sam_sorted[s_low], v2_final_sam_sorted[s_high],color='0.65',zorder=0)

# <v>
plt.subplot(2,3,5)
bovy_plot.bovy_plot(z_out,v_init,'-',gcf=True,
                   xlabel=r'$\tilde{z}$',ylabel=r'$\nu_{\mathrm{obs}}(\tilde{z})$',
                   xrange=[zmin, zmax],yrange=[vmin,vmax])
bovy_plot.bovy_plot(z_obs,v_obs,'o',overplot=True)
plt.errorbar(z_obs,v_obs,yerr=v_obs_noise,marker='None',ls='none',color=sns.color_palette()[1])
# plt.fill_between(z_out, v_final_sam_sorted[s_low], v_final_sam_sorted[s_high],color='0.65',zorder=0)
bovy_plot.bovy_plot(z_out,v_final,'-',overplot=True,zorder=0,color=final_color)

# v histogram
plt.subplot(2,3,6)
plt.hist(vz_vmock,bins=51,normed=True,range=(vzmin,vzmax), \
         histtype='step',color=constraint_color)
# xs= numpy.linspace(zmin, zmax, 201)
h,e,p= plt.hist(vz_m2m,weights=w_out,histtype='step',lw=2.,normed=True, \
                bins=31,range=(vzmin,vzmax),zorder=1,color=final_color)
plt.fill_between(0.5*(e+numpy.roll(e,1))[1:],vz_hist_sorted[s_low],vz_hist_sorted[s_high],color='0.65',zorder=0,step='mid')
plt.xlim(vzmin, vzmax)
plt.ylim(0.,0.05)
plt.xlabel(r'$v_z$')
plt.ylabel(r'$p(v_z)$')
print("Velocity dispersions: obs, fit",numpy.std(vz_vmock),\
      numpy.sqrt(numpy.sum(w_out*(vz_m2m-numpy.sum(w_out*vz_m2m)/numpy.sum(w_out))**2.)/numpy.sum(w_out)))

# stellar mass distribution
plt.subplot(2,3,2)
# plt.scatter(sfmden_star_sam,dmden_sam,s='o')
bovy_plot.bovy_plot(sfmden_star_sam,dmden_sam,'o', color=final_color, \
                    xlabel=r'$\Sigma_{\rm star}$ (Msun pc$^{-2}$)', \
                    ylabel=r'$\rho_{\rm DM}$ (Msun pc$^{-3}$)', \
                    xrange=[43.0, 48.0], yrange=[0.0, 0.03], gcf=True)
# bovy_plot.bovy_plot(sfmden_true,dmden_true,'o',overplot=True)
# print(' True values star dm =', sfmden_true, dmden_true)
if obs_data=='mock_stable' or obs_data=='mock_perturbed':
  plt.scatter(sfmden_true, dmden_true,marker='*')


# DM density distribution
plt.subplot(2,3,3)
plt.fill_between(zabs_out,sfmden_z_star_sam_sorted[s_low], \
                 sfmden_z_star_sam_sorted[s_high], \
                 color='b',alpha=0.5,zorder=0)
plt.fill_between(zabs_out,sfmden_z_dm_sam_sorted[s_low], \
                 sfmden_z_dm_sam_sorted[s_high], \
                 color='r',alpha=0.5,zorder=0)
plt.fill_between(zabs_out,sfmden_z_tot_sam_sorted[s_low], \
                 sfmden_z_tot_sam_sorted[s_high], \
                 color='0.65',alpha=0.5,zorder=0)
plt.plot(zabs_out,sfmden_z_tot_true)
plt.xlabel(r'$|z|$ (kpc)')
plt.ylabel(r'$\rho$ (Msun pc$^{-3}$)')


plt.tight_layout()
bovy_plot.bovy_end_print('sample_m2m_results.jpg')
# bring back w_sam
w_sam = copy.deepcopy(w_samallpop)



