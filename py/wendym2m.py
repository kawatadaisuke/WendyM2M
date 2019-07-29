# wendym2m.py: M2M with wendy, a 1D N-body code
import tqdm
import copy
import numpy
import wendypy
import hom2m
from itertools import chain

########################## SELF-GRAVITATING DISK TOOLS ########################
def sample_sech2(sigma,totmass,n=1):
    # compute zh based on sigma and totmass
    zh= sigma**2./totmass # twopiG = 1. in our units
    x= numpy.arctanh(2.*numpy.random.uniform(size=n)-1)*zh*2.
    v= numpy.random.normal(size=n)*sigma
    v-= numpy.mean(v) # stabilize
    m= numpy.ones_like(x)*totmass/n
    return (x,v,m)

############################### M2M FORCE-OF-CHANGE ###########################
# All defined here as the straight d constraint / d parameter (i.e., does *not*
# include things like eps, weight)

def force_of_change_weights(w_m2m,zsun_m2m,z_m2m,vz_m2m,
                            data_dicts,
                            prior,mu,w_prior,
                            h_m2m=0.02,
                            kernel=hom2m.epanechnikov_kernel,
                            delta_m2m=None,
                            xnm_m2m=1.0):
    """Computes the force of change for all of the weights"""
    fcw= numpy.zeros_like(w_m2m)
    delta_m2m_new= []
    if delta_m2m is None: delta_m2m= [None for d in data_dicts]
    for ii,data_dict in enumerate(data_dicts):
        if data_dict['type'].lower() == 'dens':
            # assuming a single population.
            tfcw, tdelta_m2m_new=\
                hom2m.force_of_change_density_weights(\
                    numpy.sum(w_m2m[:,data_dict['pops']],axis=1),
                    zsun_m2m,z_m2m,vz_m2m,
                    data_dict['zobs'],data_dict['obs'],data_dict['unc'],
                    h_m2m=h_m2m,kernel=kernel,delta_m2m=delta_m2m[ii],
                    xnm_m2m=xnm_m2m)
        elif data_dict['type'].lower() == 'v2':
            tfcw, tdelta_m2m_new=\
                hom2m.force_of_change_v2_weights(\
                    numpy.sum(w_m2m[:,data_dict['pops']],axis=1),
                    zsun_m2m,z_m2m,vz_m2m,
                    data_dict['zobs'],data_dict['obs'],data_dict['unc'],
                    h_m2m=h_m2m,kernel=kernel,deltav2_m2m=delta_m2m[ii])
        elif data_dict['type'].lower() == 'v':
            tfcw, tdelta_m2m_new=\
                hom2m.force_of_change_v_weights(\
                    numpy.sum(w_m2m[:,data_dict['pops']],axis=1),
                    zsun_m2m,z_m2m,vz_m2m,
                    data_dict['zobs'],data_dict['obs'],data_dict['unc'],
                    h_m2m=h_m2m,kernel=kernel,deltav_m2m=delta_m2m[ii])
        else:
            raise ValueError("'type' of measurement in data_dict not understood")
        fcw[:,data_dict['pops']]+= numpy.atleast_2d(tfcw).T
        # delta_m2m_new.extend(tdelta_m2m_new)
        delta_m2m_new.append(tdelta_m2m_new)
    # Add prior

    # print(' shape fcw, w, mu, w_prior =',numpy.shape(fcw), numpy.shape(w_m2m), numpy.shape(mu), numpy.shape(w_prior))
    # print(' shape fcw =',numpy.shape(hom2m.force_of_change_prior_weights(w_m2m,mu,w_prior,prior)))
    fcw+= hom2m.force_of_change_prior_weights(w_m2m,mu,w_prior,prior)
    return (fcw, delta_m2m_new)

# omega

# rewind the orbit for one step.
def rewind_zvz(z_init, vz_init, mass, omega, step):
    grewind = wendypy.nbody(
      z_init, vz_init, mass, -step, omega=omega, approx=True, nleap=1)
    z_rewind, vz_rewind = next(grewind)

    return (z_rewind, vz_rewind)

# rewind the orbit for n step.

def rewind_nstep_zvz(z_init, vz_init, mass, omega, step, nstep):
    grewind = wendypy.nbody(
      z_init, vz_init, mass, -step, omega=omega, approx=True, nleap=1)
    for ii in range(nstep):
        z_rewind, vz_rewind = next(grewind)

    return (z_rewind, vz_rewind)

# integrate forward for n step

def forward_nstep_zvz(z_init, vz_init, mass, omega, step, nstep):
    gforward = wendypy.nbody(
      z_init, vz_init, mass, -step, omega=omega, approx=True, nleap=1)
    for ii in range(nstep):
        z_forward, vz_forward = next(gforward)

    return (z_forward, vz_forward)


# Function that returns the difference in z and vz for orbits starting at the 
# same (z,vz)_init integrated in potentials with different omega
def zvzdiff(z_init, vz_init, mass, omega1, omega2, step):
    # (temporary?) way to deal with small masses
    relevant_particles_index = mass > (numpy.median(mass[mass > 10.**-9.])*10.**-6.)
    if numpy.any(mass[relevant_particles_index]
                 < (10.**-8.*numpy.median(mass[relevant_particles_index]))):
        print(
          numpy.sum(mass[relevant_particles_index]
                    < (10.**-8.*numpy.median(mass[relevant_particles_index]))))

    # integrate with wendy
    g1 = wendypy.nbody(
      z_init[relevant_particles_index], vz_init[relevant_particles_index],
      mass[relevant_particles_index], step, omega=omega1, approx=True, nleap=1)
    z_next1, vz_next1 = next(g1)
    dz1 = numpy.zeros_like(z_init)
    dvz1 = numpy.zeros_like(z_init)
    dz1[relevant_particles_index] = z_next1-z_init[relevant_particles_index]
    dvz1[relevant_particles_index] = vz_next1-vz_init[relevant_particles_index]
    g2 = wendypy.nbody(
      z_init[relevant_particles_index], vz_init[relevant_particles_index],
      mass[relevant_particles_index], step, omega=omega2, approx=True, nleap=1)
    z_next2, vz_next2 = next(g2)
    dz2 = numpy.zeros_like(z_init)
    dvz2 = numpy.zeros_like(z_init)
    dz2[relevant_particles_index] = z_next2-z_init[relevant_particles_index]
    dvz2[relevant_particles_index] = vz_next2-vz_init[relevant_particles_index]
    
    return (dz2-dz1, dvz2-dvz1)
    
def force_of_change_omega(w_m2m,zsun_m2m,omega_m2m,
                          z_m2m,vz_m2m,z_prev,vz_prev,
                          step,data_dicts,
                          delta_m2m,
                          h_m2m=0.02,kernel=hom2m.epanechnikov_kernel,
                          delta_omega=0.3, xnm_m2m=1.0):
    """Compute the force of change by direct finite difference 
    of the objective function"""
    mass = numpy.sum(w_m2m, axis=1)
    dz, dvz = zvzdiff(
      z_prev, vz_prev, mass, omega_m2m, omega_m2m+delta_omega, step)

    fcw, delta_m2m_do = force_of_change_weights(\
        w_m2m,zsun_m2m,z_m2m+dz,vz_m2m+dvz,
        data_dicts,
        'entropy',0.,1., # weights prior doesn't matter, so set to zero
        h_m2m=h_m2m,kernel=kernel,
        delta_m2m=delta_m2m, xnm_m2m=xnm_m2m)
#    return -numpy.nansum(\
#        delta_m2m*(delta_m2m_do-delta_m2m)/dens_obs_noise
#        +deltav2_m2m*(deltav2_m2m_do-deltav2_m2m)/densv2_obs_noise)\
#        /delta_omega

    return -numpy.nansum(\
        delta_m2m[0]*(delta_m2m_do[0]-delta_m2m[0])
        +delta_m2m[1]*(delta_m2m_do[1]-delta_m2m[1]))\
        /delta_omega

# Xnm
def force_of_change_xnm(w_m2m,zsun_m2m,z_m2m,vz_m2m,
                        data_dicts,
                        delta_m2m,h_m2m=0.02,
                        kernel=hom2m.epanechnikov_kernel,
                        xnm_m2m=1.0):
    """Computes the force of change for Xnm"""
    z_obs = data_dicts['type' == 'dens']['zobs']
    dens_obs_noise = data_dicts['type' == 'dens']['unc']
    Wij = numpy.zeros((len(z_obs), len(z_m2m)))
    mdens_m2m = numpy.zeros(len(z_obs))
    for jj,zo in enumerate(z_obs):
        Wij[jj] = kernel(numpy.fabs(zo-z_m2m+zsun_m2m), h_m2m)
        mdens_m2m[jj] = numpy.nansum(w_m2m*Wij[jj])

    return (-numpy.nansum(mdens_m2m*delta_m2m[0]/dens_obs_noise))

################################ M2M OPTIMIZATION #############################
def parse_data_dict(data_dicts):
    """
    NAME:
       parse_data_dict
    PURPOSE:
       parse the data_dict input to M2M routines
    INPUT:
       data_dicts - list of data_dicts
    OUTPUT:
       cleaned-up version of data_dicts
    HISTORY:
       2017-07-20 - Written - Bovy (UofT)
    """
    for data_dict in data_dicts:
        if isinstance(data_dict['pops'],int):
            data_dict['pops']= [data_dict['pops']]
    return data_dict
  

def fit_m2m(w_init,z_init,vz_init,
            omega_m2m,zsun_m2m,
            data_dicts,
            step=0.001,nstep=1000,
            eps=0.1,mu=1.,prior='entropy',w_prior=None,
            kernel=hom2m.epanechnikov_kernel,
            kernel_deriv=hom2m.epanechnikov_kernel_deriv,
            h_m2m=0.02,
            npop=1,
            smooth=None,st96smooth=False,
            output_wevolution=False,
            output_zvzevolution = False, 
            fit_zsun=False,fit_omega=False,
            skipomega=10,delta_omega=0.3,
            number_density=False, xnm_m2m=1.0, skipxnm=10, fit_xnm=False
            ):
    """
    NAME:
       fit_m2m
    PURPOSE:
       Run M2M optimization for wendy M2M
    INPUT:
       w_init - initial weights [N] or [N,npop]
       z_init - initial z [N]
       vz_init - initial vz (rad) [N]
       omega_m2m - background potential parameter omega, if None no background
       zsun_m2m - Sun's height above the plane [N]
       data_dicts - list of dictionaries that hold the data, these are described in more detail below
       step= stepsize of orbit integration
       nstep= number of steps to integrate the orbits for
       eps= M2M epsilon parameter (can be array when fitting zsun, omega; in that case eps[0] = eps_weights, eps[1] = eps_zsun, eps[1 or 2 based on fit_zsun] = eps_omega)
       mu= M2M entropy parameter mu
       prior= ('entropy' or 'gamma')
       w_prior= (None) prior weights (if None, equal to w_init)
       fit_zsun= (False) if True, also optimize zsun
       fit_omega= (False) if True, also optimize omega
       skipomega= only update omega every skipomega steps
       delta_omega= (0.3) difference in omega to use to compute derivative of objective function wrt omega
       kernel= a smoothing kernel
       kernel_deriv= the derivative of the smoothing kernel
       h_m2m= kernel size parameter for computing the observables
       npop= (1) number of theoretical populations
       smooth= smoothing parameter alpha (None for no smoothing)
       st96smooth= (False) if True, smooth the constraints (Syer & Tremaine 1996), if False, smooth the objective function and its derivative (Dehnen 2000)
       output_wevolution= if set to an integer, return the time evolution of this many randomly selected weights
       output_zvzevolution= if set to an integer, return the time evolution
                of this many randomly selected weights only when
                output_wevolution is True
       number_density = (False) if True, observed density is number density and density calculation requires xnm
       xnm_m2m = (1.0) initial value of xnm: number_density/mass_density [1], assuming a single population
       skipxnm = only update Xnm every skipxnm steps
       fit_xnm = (False) if True, also optimise xnm
    DATA DICTIONARIES:
       The data dictionaries have the following form:
           'type': type of measurement: 'dens', 'v2'
           'pops': the theoretical populations included in this measurement; 
                   single number or list
           'zobs': vertical height of the observation
           'zrange': width of vertical bin relative to some fiducial value (used to scale h_m2m, which should therefore be appropriate for the fiducial value)
           'obs': the actual observation
           'unc': the uncertainty in the observation
       of these, zobs, obs, and unc can be arrays for mulitple measurements
    OUTPUT:
       (w_out,[zsun_out, [omega_out, [xnm_out]],z_m2m,vz_m2m,Q_out,[wevol,rndindx]) - 
              (output weights [N],
              [Solar offset [nstep] optional],
              [omega [nstep] optional when fit_omega],
              z_m2m [N] final z,
              vz_m2m [N] final vz,
              objective function as a function of time [nstep],
              [weight evolution for randomly selected weights,index of random weights])
    HISTORY:
       2017-07-20 - Started from hom2m.fit_m2m - Bovy (UofT)
       2018-10-16 - add external potential using omega_m2m - Kawata (MSSL/UCL)
       2018-10-29 - add xnm=number_ensity/mass_density - Kawata (MSSL/UCL)
    """
    if len(w_init.shape) == 1:
        w_out= numpy.empty((len(w_init),npop))
        w_out[:,:]= numpy.tile(copy.deepcopy(w_init),(npop,1)).T
    else:
        w_out= copy.deepcopy(w_init)
    zsun_out= numpy.empty(nstep)
    omega_out= numpy.empty(nstep)
    if number_density:
        xnm_out= numpy.empty(nstep)
    else:
        xnm_out= numpy.ones(nstep)
        xnm_m2m = 1.0
    if w_prior is None:
        if len(w_init.shape) == 1:
            w_prior= numpy.empty((len(w_init),npop))
            w_prior[:,:]= numpy.tile(copy.deepcopy(w_init),(npop,1)).T
        else:
            w_piror= copy.deepcopy(w_init)
    else:
        if len(w_prior.shape) == 1:
            w_prior = numpy.tile(w_prior, (npop, 1)).T
    # Parse data_dict
    data_dict= parse_data_dict(data_dicts)
    # Parse eps
    if isinstance(eps,float):
        eps= [eps]
        if fit_zsun: eps.append(eps[0])
        if fit_omega: eps.append(eps[0])
        if fit_xnm: eps.append(eps[0])
    Q_out= []
    if output_wevolution:
        rndindx= numpy.random.permutation(len(w_out))[:output_wevolution]
        wevol= numpy.zeros((output_wevolution,npop,nstep))
        if output_zvzevolution:
            zevol = numpy.zeros((output_wevolution,nstep))
            vzevol = numpy.zeros((output_wevolution,nstep))
    # Compute force of change for first iteration
    fcw, delta_m2m_new = \
        force_of_change_weights(w_out,zsun_m2m,z_init,vz_init,
                                data_dicts,prior,mu,w_prior,
                                h_m2m=h_m2m,kernel=kernel,
                                xnm_m2m=xnm_m2m)
    fcw*= w_out
    fcz= 0.
    if fit_zsun:
        fcz= force_of_change_zsun(w_init,zsun_m2m,z_init,vz_init,
                                  z_obs,dens_obs_noise,delta_m2m_new,
                                  densv2_obs_noise,deltav2_m2m_new,
                                  kernel=kernel,kernel_deriv=kernel_deriv,
                                  h_m2m=h_m2m)
    fcxnm = 0.0
    if fit_xnm:
        fcxnm = force_of_change_xnm(w_init, zsun_m2m, z_init, vz_init,
                                    data_dicts, delta_m2m_new, h_m2m=h_m2m,
                                    kernel=kernel, xnm_m2m=xnm_m2m)
    if not smooth is None:
        delta_m2m= delta_m2m_new
    else:
        delta_m2m= [None for d in data_dicts]
    if not smooth is None and not st96smooth:
        Q= [d**2 for d in delta_m2m**2.]
    # setup skipomega omega counter and prev. (z,vz) for F(omega)
    xcounter= skipxnm-1 # Causes F(Xnm) to be computed in the 1st step    
    ocounter= skipomega-1 # Causes F(omega) to be computed in the 1st step
    # Rewind for first step
    mass = numpy.sum(w_out, axis=1)
    z_prev, vz_prev = rewind_zvz(z_init, vz_init, mass, omega_m2m, step)
    z_m2m, vz_m2m= z_init, vz_init
    for ii in range(nstep):
        # Update weights first
        if True:
            w_out+= eps[0]*step*fcw
            w_out[w_out < 10.**-16.]= 10.**-16.
        # then zsun
        if fit_zsun: 
            zsun_m2m+= eps[1]*step*fcz 
            zsun_out[ii]= zsun_m2m
        # then xnm
        if fit_xnm and xcounter == skipxnm:
            dxnm = eps[1+fit_zsun+fit_omega]*step*fcxnm
            max_dxnm = xnm_m2m/10.0
            if numpy.fabs(dxnm) > max_dxnm:
                dxnm = max_dxnm*numpy.sign(dxnm)
            xnm_m2m += dxnm
            # print(ii,' step Xnm=',xnm_m2m, eps[1+fit_zsun+fit_omega])
            xcounter = 0
        # then omega (skipped in the first step, so undeclared vars okay)
        if fit_omega and ocounter == skipomega:
            domega= eps[1+fit_zsun]*step*skipomega*fco
            # max_domega= delta_omega/30.
            max_domega= omega_m2m/10.0
            if numpy.fabs(domega) > max_domega:
                domega= max_domega*numpy.sign(domega)
            omega_m2m+= domega
            # Keep (z,vz) the same in new potential
            # A_now, phi_now= zvz_to_Aphi(z_m2m,vz_m2m,omega_m2m)
            ocounter= 0
        # (Store objective function)
        if not smooth is None and st96smooth:
            Q_out.append([d**2. for d in delta_m2m])
        elif not smooth is None:
            Q_out.append(copy.deepcopy(Q))
        else:
            Q_out.append([d**2. for d in list(chain.from_iterable(delta_m2m_new))])
        # Then update the dynamics
        mass= numpy.sum(w_out,axis=1)
        # (temporary?) way to deal with small masses
        relevant_particles_index= mass > (numpy.median(mass[mass > 10.**-9.])*10.**-6.)
        if numpy.any(mass[relevant_particles_index] < (10.**-8.*numpy.median(mass[relevant_particles_index]))):
            print(numpy.sum(mass[relevant_particles_index] < (10.**-8.*numpy.median(mass[relevant_particles_index]))))
        # g= wendypy.nbody(z_m2m[relevant_particles_index],
        #               vz_m2m[relevant_particles_index],
        #               mass[relevant_particles_index],
        #               step, omega=omega_m2m, maxcoll=10000000)
        g= wendypy.nbody(z_m2m[relevant_particles_index],
                       vz_m2m[relevant_particles_index],
                       mass[relevant_particles_index],
                       step, omega=omega_m2m, approx=True, nleap=1)
        tz_m2m, tvz_m2m= next(g)
        z_m2m[relevant_particles_index]= tz_m2m
        vz_m2m[relevant_particles_index]= tvz_m2m
        z_m2m-= numpy.sum(mass*z_m2m)/numpy.sum(mass)
        vz_m2m-= numpy.sum(mass*vz_m2m)/numpy.sum(mass)
        # Compute force of change
        if smooth is None or not st96smooth:
            # Turn these off
            tdelta_m2m= None
        else:
            tdelta_m2m= delta_m2m
        fcw_new, delta_m2m_new = \
            force_of_change_weights(w_out,zsun_m2m,z_m2m,vz_m2m,
                                    data_dicts,prior,mu,w_prior,
                                    h_m2m=h_m2m,kernel=kernel,
                                    delta_m2m=tdelta_m2m, xnm_m2m=xnm_m2m)
        fcw_new*= w_out
        if fit_zsun:
            if smooth is None or not st96smooth:
                tdelta_m2m= delta_m2m_new
            fcz_new= force_of_change_zsun(w_out,zsun_m2m,z_m2m,vz_m2m,
                                          data_dicts,tdelta_m2m,
                                          kernel=kernel,
                                          kernel_deriv=kernel_deriv,
                                          h_m2m=h_m2m)
        if fit_xnm:
            # Update Xnm in this step?
            xnm_out[ii] = xnm_m2m          
            xcounter += 1
            if xcounter == skipxnm:
                if smooth is None or not st96smooth:
                    tdelta_m2m = delta_m2m_new
                fcxnm_new = force_of_change_xnm(
                  w_out, zsun_m2m, z_m2m, vz_m2m, data_dicts,
                  tdelta_m2m, h_m2m=h_m2m, kernel=kernel, xnm_m2m=xnm_m2m)
        if fit_omega:
            omega_out[ii]= omega_m2m
            # Update omega in this step?
            ocounter+= 1
            if ocounter == skipomega:
                if not fit_zsun and (smooth is None or not st96smooth):
                    tdelta_m2m= delta_m2m_new
                    # tdeltav2_m2m= deltav2_m2m_new
                fco_new= force_of_change_omega(w_out,zsun_m2m,omega_m2m,
                                               z_m2m,vz_m2m,z_prev,vz_prev,
                                               step*skipomega,
                                               data_dicts, tdelta_m2m,
                                               h_m2m=h_m2m,kernel=kernel,
                                               delta_omega=delta_omega,
                                               xnm_m2m=xnm_m2m)
                z_prev= copy.copy(z_m2m)
                vz_prev= copy.copy(vz_m2m)
        # Increment smoothing
        if not smooth is None and st96smooth:
            delta_m2m= [d+step*smooth*(dn-d) 
                        for d,dn in zip(list(chain.from_iterable(delta_m2m)),
                                        list(chain.from_iterable(delta_m2m_new)))]
            fcw= fcw_new
            if fit_zsun: fcz= fcz_new
            if fit_omega and ocounter == skipomega: fco= fco_new
            if fit_xnm and xcounter == skipxnm: fcxnm = fcxnm_new
        elif not smooth is None:
            Q_new= [d**2. for d in delta_m2m_new]
            Q= [q+step*smooth*(qn-q) for q,qn in zip(Q,Q_new)]
            fcw+= step*smooth*(fcw_new-fcw)
            if fit_zsun: fcz+= step*smooth*(fcz_new-fcz)
            if fit_xnm and xcounter == skipxnm:
                fcxnm += step*smooth*(fcxnm-fcxnm_new)
            if fit_omega and ocounter == skipomega:
                fco+= step*skipomega*smooth*(fco_new-fco)
        else:
            fcw= fcw_new
            if fit_zsun: fcz= fcz_new
            if fit_xnm and xcounter == skipxnm: fcxnm = fcxnm_new
            if fit_omega and ocounter == skipomega: fco= fco_new
        # Record random weights if requested
        if output_wevolution:
            wevol[:,:,ii]= w_out[rndindx]
            if output_zvzevolution:
                zevol[:, ii] = z_m2m[rndindx]
                vzevol[:, ii] = vz_m2m[rndindx]
    out= (w_out,)
    if fit_zsun: out= out+(zsun_out,)
    if fit_omega:
        out= out+(omega_out,)
    if fit_xnm:
        out= out+(xnm_out,)
    out= out+(z_m2m,vz_m2m,)
    out= out+(numpy.array(Q_out),)
    if output_wevolution:
        out= out+(wevol,rndindx,)
        if output_zvzevolution:
            out = out+(zevol,)
            out = out+(vzevol,)
    return out
  
def sample_m2m(nsamples,
               w_init,z_init,vz_init,
               omega_m2m,zsun_m2m,
               data_dicts, **kwargs):
    """
    NAME:
       sample_m2m
    PURPOSE:
       Sample parameters using M2M optimization for the weights and Metropolis-Hastings for the other parameters
    INPUT:
       nsamples - number of samples from the ~PDF
       fix_weights= (False) if True, don't sample the weights

       zsun parameters:
          sig_zsun= (0.005) if sampling zsun (fit_zsun=True), proposal stepsize for steps in zsun
          nmh_zsun= (20) number of MH steps to do for zsun for each weights sample
          nstep_zsun= (500) number of steps to average the likelihood over for zsun MH

       xnm parameters:
          sig_xnm= (0.005) if sampling zsun (fit_zsun=True), proposal stepsize for steps in Xnm
          nmh_xnm= (20) number of MH steps to do for zsun for each weights sample
          nstep_xnm= (500) number of steps to average the likelihood over for Xnm MH

       omega parameters:
          sig_omega= (0.2) if sampling omega (fit_omega=True), proposal stepsize for steps in omega
          nmh_omega= (20) number of MH steps to do for omega for each weights sample
          nstep_omega= (500) number of steps to average the likelihood over for omega MH; also the number of steps taken to change omega adiabatically
          nstepadfac_omega= (10) use nstepadfac_omega x nstep_omega steps to adiabatically change the frequency to the proposed value

       Rest of the parameters are the same as for fit_m2m
    OUTPUT:
       (w_out,[zsun_out],Q_out,z,vz) - 
               (output weights [nsamples,N],
               [Solar offset [nsamples],
               objective function [nsamples,nobs],
               positions at the final step of each sample [nsamples,N],
               velocities at the final step of each sample [nsamples,N])
    HISTORY:
       2019-04-29 - Copied from Jo Bovy's simple-m2m/py/hom2m.py by D. Kawata (MSSL, UCL)
    """
    nw= len(w_init)
    npop = kwargs.get('npop',1)
    w_out= numpy.empty((nsamples, nw, npop))
    z_out= numpy.empty((nsamples, nw))
    vz_out= numpy.empty_like(z_out)
    step= kwargs.get('step',0.001)
    eps= kwargs.get('eps',0.001)
    nstep= kwargs.get('nstep',1000)
    fix_weights= kwargs.pop('fix_weights',False)
    # turn off w_evolution
    kwargs['output_wevolution']= False
    # zsun
    fit_zsun= kwargs.get('fit_zsun',False)
    kwargs['fit_zsun']= False # Turn off for weights fits
    sig_zsun= kwargs.pop('sig_zsun',0.005)
    nmh_zsun= kwargs.pop('nmh_zsun',20)
    nstep_zsun= kwargs.pop('nstep_zsun',499*fit_zsun+1)
    if fit_zsun: 
        zsun_out= numpy.empty((nsamples))
        nacc_zsun= 0
    # xnm
    fit_xnm= kwargs.get('fit_xnm',False)
    kwargs['fit_xnm']= False # Turn off for weights fits
    sig_xnm= kwargs.pop('sig_xnm',0.005)
    nmh_xnm= kwargs.pop('nmh_xnm',20)
    nstep_xnm= kwargs.pop('nstep_xnm',nstep_zsun*fit_zsun \
                          +500*(1-fit_zsun))
    if fit_xnm:
        xnm_out= numpy.empty((nsamples))
        nacc_xnm= 0
    # get xnm
    xnm_m2m= kwargs.get('xnm_m2m',1.0)              
    # omega
    fit_omega= kwargs.get('fit_omega',False)
    kwargs['fit_omega']= False # Turn off for weights fits
    sig_omega= kwargs.pop('sig_omega',0.005)
    nmh_omega= kwargs.pop('nmh_omega',20)
    nstep_omega= kwargs.pop('nstep_omega',numpy.maximum(nstep_zsun*fit_zsun \
                            +500*(1-fit_zsun),
                            nstep_xnm*fit_xnm+500*(1-fit_xnm)))
    nstepadfac_omega= kwargs.pop('nstepadfac_omega',10)
    if fit_omega: 
        omega_out= numpy.empty((nsamples))
        nacc_omega= 0
    # Copy some kwargs that we need to re-use
    nout = 0

    for ii,data_dict in enumerate(data_dicts):
        if data_dict['type'].lower() == 'dens':
          dens_obs = copy.deepcopy(data_dict['obs'])
          nout += len(dens_obs)
        elif data_dict['type'].lower() == 'v2':
          v2_obs = copy.deepcopy(data_dict['obs'])
          nout += len(v2_obs)
        elif data_dict['type'].lower() == 'v':
          v_obs = copy.deepcopy(data_dict['obs'])
          nout += len(v_obs)
        else:
            raise ValueError("'type' of measurement in data_dict in sample_m2m not understood")
    Q_out= numpy.empty((nsamples, nout))
    
    # Setup orbits
    # A_now, phi_now= zvz_to_Aphi(z_init,vz_init,omega_m2m)
    z_m2m= z_init
    vz_m2m= vz_init
    for ii in tqdm.tqdm(range(nsamples)):
        if not fix_weights:
            # Draw new observations
            for jj,data_dict in enumerate(data_dicts):
              if data_dict['type'].lower() == 'dens':
                data_dicts[jj]['obs'] = dens_obs\
                  +numpy.random.normal(size=len(dens_obs))*data_dict['unc']
              elif data_dict['type'].lower() == 'v2':
                data_dicts[jj]['obs'] = v2_obs\
                  +numpy.random.normal(size=len(v2_obs))*data_dict['unc']
              elif data_dict['type'].lower() == 'v':
                # assuming v2 constraints as well
                data_dicts[jj]['obs'] = v_obs\
                  +numpy.random.normal(size=len(v_obs))*data_dict['unc']
              else:
                raise ValueError( \
                  "'type' of measurement in data_dict in sample_m2m not understood")
            tout= fit_m2m(w_init,z_init,vz_init,omega_m2m,zsun_m2m, \
                          data_dicts, **kwargs)
            # Keep track of orbits
            # phi_now+= omega_m2m*kwargs.get('nstep',1000) \
            #    *kwargs.get('step',0.001)
            # z_m2m, vz_m2m= Aphi_to_zvz(A_now,phi_now,omega_m2m)
            # Let's not do this for now
            # z_m2m = z_init
            # vz_m2m = vz_m2m
            # Need to switch back to original data
            for jj,data_dict in enumerate(data_dicts):
              if data_dict['type'].lower() == 'dens':
                data_dicts['type' == 'dens']['obs'] = dens_obs
              elif data_dict['type'].lower() == 'v2':
                data_dicts['type' == 'v2']['obs'] = v2_obs
              elif data_dict['type'].lower() == 'v':
                data_dicts['type' == 'v']['obs'] = v_obs
              else:
                raise ValueError( \
                  "'type' of measurement in data_dict in sample_m2m not understood")
        else:
            tout= [w_init]
        # Compute average chi^2 for initial zsun
        if fit_zsun:
            tnstep= nstep_zsun
        elif fit_xnm:
            tnstep= nstep_xnm
        else:
            tnstep= nstep_omega
        kwargs['nstep']= tnstep
        kwargs['eps']= 0. # Don't change weights
        # keep the original z and vz
        z0_m2m = copy.deepcopy(z_m2m)
        vz0_m2m = copy.deepcopy(vz_m2m)
        dum_wout, dum_z, dum_vz, dum_Q = fit_m2m( \
                     tout[0],z_m2m,vz_m2m,omega_m2m,zsun_m2m, \
                     data_dicts, **kwargs)
        kwargs['nstep']= nstep
        kwargs['eps']= eps
        tQ= numpy.mean(dum_Q, axis=0)
        # Keep track of orbits
        z_m2m = dum_z
        vz_m2m = dum_vz
        if fit_zsun:
            # Rewind orbit, so we use same part for all zsun/omega
            # phi_now-= omega_m2m*nstep_zsun*kwargs.get('step',0.001)
            # Rewind by -nstep_zsun
            mass = numpy.sum(tout[0], axis=1)
            z_m2m, vz_m2m = rewind_nstep_zvz(z0_m2m, vz0_m2m, mass,
                                               omega_m2m, step, nstep)
            kwargs['nstep']= nstep_zsun
            kwargs['eps']= 0. # Don't change weights
            for jj in range(nmh_zsun):
                # Do a MH step
                zsun_new= zsun_m2m+numpy.random.normal()*sig_zsun
                dum_wout, dum_z, dum_vz, dum_Q = fit_m2m(tout[0], \
                   z_m2m,vz_m2m,omega_m2m,zsun_new,data_dicts, **kwargs)
                acc= (numpy.nansum(tQ)
                      -numpy.mean(numpy.nansum(dum_Q, axis=1)))/2.
                if acc > numpy.log(numpy.random.uniform()):
                    zsun_m2m= zsun_new
                    tQ= numpy.mean(dum_Q, axis=0)
                    nacc_zsun+= 1
            kwargs['nstep']= nstep
            kwargs['eps']= eps
            zsun_out[ii]= zsun_m2m
            # update orbit
            z_m2m = dum_z
            vz_m2m = dum_vz
        if fit_zsun and nstepzsun != nstep_xnm:
            # Need to compute average obj. function for nstep_omega
            kwargs['nstep']= nstep_xnm
            kwargs['eps']= 0. # Don't change weights
            dum_wout, dum_z, dum_vz, dum_Q = fit_m2m(tout[0], \
              z_m2m,vz_m2m,omega_m2m,zsun_m2m,data_discts, **kwargs)
            kwargs['nstep']= nstep
            kwargs['eps']= eps
            tQ= numpy.mean(dum_Q, axis=0)
            # Keep track of orbits
            z_m2m = dum_z
            vz_m2m = dum_vz
        if fit_xnm:
            # Rewind by -nstep_zsun
            mass = numpy.sum(tout[0], axis=1)
            z_m2m, vz_m2m = rewind_nstep_zvz(z0_m2m, vz0_m2m, mass,
                                               omega_m2m, step, nstep_xnm)
            kwargs['nstep']= nstep_xnm
            kwargs['eps']= 0. # Don't change weights
            for jj in range(nmh_xnm):
                # Do a MH step
                xnm_new= xnm_m2m+numpy.random.normal()*sig_xnm
                kwargs['xnm_m2m']= xnm_new
                dum_wout, dum_z, dum_vz, dum_Q = fit_m2m(tout[0], \
                   z_m2m,vz_m2m,omega_m2m,zsun_m2m,data_dicts, **kwargs)
                acc= (numpy.nansum(tQ)
                      -numpy.mean(numpy.nansum(dum_Q, axis=1)))/2.
                if acc > numpy.log(numpy.random.uniform()):
                    xnm_m2m= xnm_new
                    tQ= numpy.mean(dum_Q, axis=0)
                    nacc_xnm+= 1
            kwargs['nstep']= nstep
            kwargs['eps']= eps
            kwargs['xnm_m2m']= xnm_new            
            xnm_out[ii]= xnm_m2m
            # update orbit
            z_m2m = dum_z
            vz_m2m = dum_vz
        if (fit_zsun and nstepzsun != nstep_omega) or \
           (fit_xnm and nstep_xnm !=nstep_omega):
            # Need to compute average obj. function for nstep_omega
            kwargs['nstep']= nstep_omega
            kwargs['eps']= 0. # Don't change weights
            dum_wout, dum_z, dum_vz, dum_Q = fit_m2m(tout[0], \
              z_m2m,vz_m2m,omega_m2m,zsun_m2m,data_discts, **kwargs)
            kwargs['nstep']= nstep
            kwargs['eps']= eps
            tQ= numpy.mean(dum_Q, axis=0)
            # Keep track of orbits
            z_m2m = dum_z
            vz_m2m = dum_vz
        if fit_omega:
            kwargs['nstep']= nstep_omega
            kwargs['eps']= 0. # Don't change weights
            for jj in range(nmh_omega):
                # get mass
                mass = numpy.sum(tout[0], axis=1)
                # Do a MH step
                omega_new= omega_m2m+numpy.random.normal()*sig_omega
                # Slowly change the orbits from omega to omega_new, by 
                # integrating backward
                z_cur= copy.copy(z_m2m)
                vz_cur= copy.copy(vz_m2m)
                for kk in range(nstep_omega*nstepadfac_omega):
                    omega_cur= omega_m2m+(omega_new-omega_m2m)\
                        *kk/float(nstep_omega*nstepadfac_omega-1)
                    z_cur, vz_cur = rewind_zvz(z_cur, vz_cur, mass, omega_cur,
                                               step)
                # and forward again!
                z_cur, vz_cur = forward_nstep_zvz(z_cur, vz_cur, mass,
                                omega_cur, step,
                                nstep_omega*nstepadfac_omega)
                dum_wout, dum_z, dum_vz, dum_Q = fit_m2m(tout[0], \
                   z_cur,vz_cur,omega_new,zsun_m2m,data_dicts, **kwargs)
                acc= (numpy.nansum(tQ)
                      -numpy.mean(numpy.nansum(dum_Q, axis=1)))/2.
                if acc > numpy.log(numpy.random.uniform()):
                    omega_m2m= omega_new
                    tQ= numpy.mean(dum_Q, axis=0)                    
                    nacc_omega+= 1
            # Update phase-space positions
            z_m2m = dum_z
            vz_m2m= dum_vz
            kwargs['nstep']= nstep
            kwargs['eps']= eps
            omega_out[ii]= omega_m2m
        w_out[ii]= tout[0]
        Q_out[ii]= tQ
        z_out[ii]= z_m2m
        vz_out[ii]= vz_m2m
    out= (w_out,)
    if fit_zsun: out= out+(zsun_out,)
    if fit_xnm: out= out+(xnm_out,)    
    if fit_omega: out= out+(omega_out,)
    out= out+(Q_out,z_out,vz_out,)
    if fit_zsun: print("MH acceptance ratio for zsun was %.2f" \
                           % (nacc_zsun/float(nmh_zsun*nsamples)))
    if fit_xnm: print("MH acceptance ratio for xnm was %.2f" \
                           % (nacc_xnm/float(nmh_xnm*nsamples)))
    if fit_omega: print("MH acceptance ratio for omega was %.2f" \
                            % (nacc_omega/float(nmh_omega*nsamples)))
    return out
