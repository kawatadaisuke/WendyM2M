# wendypy.py: 1D N-body aproximate solver code with python
import numpy
import copy

def nbody(x,v,m,dt,twopiG=1.,omega=None,approx=True,nleap=None,
          maxcoll=100000,warn_maxcoll=False,
          full_output=False):
    """
    NAME:
       nbody
    PURPOSE:
       run an N-body simulation in 1D
    INPUT:
       x - positions [N]
       v - velocities [N]
       m - masses [N]
       dt - time step
       twopiG= (1.) value of 2 \pi G
       omega= (None) if set, frequency of external harmonic oscillator
       approx= (True) if True, solve the dynamics approximately using leapfrog with exact force evaluations
       nleap= (None) when approx == True, number of leapfrog steps for each dt
       maxcoll= (100000) maximum number of collisions to allow in one time step
       warn_maxcoll= (False) if True, do not raise an error when the maximum number of collisions is exceeded, but instead raise a warning and continue after re-arranging the particles
       full_output= (False) if True, also yield diagnostic information: (a) total number of collisions processed up to this iteration (cumulative; only for exact algorithm), (b) time elapsed resolving collisions if approx is False and for integrating the system if approx is True in just this iteration  (*not* cumulative)
    OUTPUT:
       Generator: each iteration returns (x,v) at equally-spaced time intervals
       + diagnostic info if full_output
    HISTORY:
       2019-04-26 - copy from Bovy's wendy code by Kawata (MSSL, UCL)
       2017-04-24 - Written - Bovy (UofT/CCA)
       2017-05-23 - Added omega - Bovy (UofT/CCA)
    """

    for item in _nbody_approx(x, v, m, dt, nleap, omega=omega, twopiG=twopiG):
      yield item

def _nbody_approx(x,v,m,dt,nleap,omega=None,twopiG=1.):
    """
    NAME:
       _nbody_approx
    PURPOSE:
       run an N-body simulation in 1D, using approximate integration w/ exact forces
    INPUT:
       x - positions [N]
       v - velocities [N]
       m - masses [N]
       dt - output time step
       nleap - number of leapfrog steps / output time step
       omega= (None) if set, frequency of external harmonic oscillator
       twopiG= (1.) value of 2 \pi G
    OUTPUT:
       Generator: each iteration returns (x,v) at equally-spaced time intervals
    HISTORY:
       2019-04-26 - copy and modified by Kawata (MSSL/UCL)
       2017-06-03 - Written - Bovy (UofT/CCA)
    """
    # run approximate solver
    if omega is None:
        omega2= 0.0
    else:
        omega2= omega**2.
    m = twopiG*copy.copy(m)
    cumulmass= numpy.zeros(len(x))
    revcumulmass= numpy.zeros(len(x))
    # Leapfrog integration timestep
    dt_leap= dt/nleap
    # initial leapfrog integration
    N = len(x)
    # drift, kick drift
    dt_leap_hf = 0.5*dt_leap
    mtot = numpy.sum(m)
    while True:
      for ii in range(nleap):
        xhf = x+dt_leap_hf*v
        a = _nbody_force(N, xhf, m, mtot, omega2)
        v = v+dt_leap*a
        x = xhf+dt_leap_hf*v
      
      yield (x,v)

def _nbody_force(N, x, m, mtot, omega2):
  # sort
  indx = numpy.argsort(x, kind='stable')
  mord = m[indx]
  cumulmass = numpy.zeros(N)
  cumulmass[indx] = numpy.cumsum(mord)
  revcumulmass = mtot-cumulmass
  cumulmass -= m
  # for ii in range(10):
  #  print('i, x, cm, rcm=', indx[ii], x[indx[ii]], cumulmass[indx[ii]],
  #        revcumulmass[indx[ii]])

  a = revcumulmass-cumulmass-omega2*x

  return a
