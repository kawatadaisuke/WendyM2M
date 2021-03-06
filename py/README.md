
# Wendy M2M

Made-to-measure (M2M) algorithm with one dimensional self-gravity. 


## AUTHORS

Daisuke Kawata - d dot kawata at ucl dot ac dot uk 

Jo Bovy - bovy at astro dot utoronto dot ca

Jason Hunt - jason dot hunt at utoronto dot ca

Morgan Bennett - bennett at astro dot utoronto dot ca

## How To

### Analysis of the Gaia DR2 data

Running M2M modelling, Unit normalisation: 2 pi G= L (kpc) = V (km/s) = 1

> python run_WendyM2M_BB19_GaiaDR2.py

MCMC sampling

> python run_sample_WendyM2M_BB19_GaiaDR2.py

### Generating mock data of a perturbed disk and applying M2M

run_WendyM2M_perturbed.py

#### notebook version

WendyM2M_XomegaXnm_rhov2obs_perturbed.ipynb

## Code

## WendyM2M_readfile.ipynb

 Notebook for testing to read the target disk file and run M2M.

## Generate_disk.ipynb

 Notebook for generating a target disk and save it in a file 'target_disk.h5'.

## wendypy_test.py

 Notebook for testing wendypy.py.

## wendypy.py

 1D approximate nbody version of Wendy in python.

## run_WendyM2M_BB19_GaiaDR2.py

 Python code to run WendyM2M for the Bennett & Bovy (2019) data.

## run_sample_WendyM2M_BB19_GaiaDR2.py

 Python code to run MCMC sampling from the result of run_WendyM2M_BB19_GaiaDR2.py.

## run_WendyM2M_sample_XomegaXnm_rhov2obs_perturbed.py

 Python code to run the same model as WendyM2M_sample_XomegaXnm_rhov2obs_perturbed for both omega and Xnm fitting. 

## WendyM2M_sample_XomegaXnm_rhov2obs_perturbed.ipynb

 Notebook for MCMC sample to evaluate uncertainties of WendyM2M modelling. 

## WendyM2M_XomegaXnm_rhov2obs_perturbed.ipynb

 Notebook for WendyM2M with an external potential taking into account number/mass density ratio, Xnm, and applying to a perturbed disk. The observational constraints are density and the square of velocity only. 

## WendyM2M_XomegaXnm_vobs_perturbed.ipynb

 Notebook for WendyM2M with an external potential taking into account number/mass density ratio, Xnm, and applying to a perturbed disk. The observational constraints include the mean of v. 


## WendyM2M_Xnm_perturbed.ipynb

 Notebook for WendyM2M with an external potential taking into account number/mass density ratio, Xnm, and applying to a perturbed disk.

## WendyM2M_Xnm.ipynb

 Notebook for WendyM2M with an external potential taking into account number/mass density ratio, Xnm

## WendyM2M_XOmega.ipynb

 Notebook for WendyM2M with a background potential and fitting omega. 


## WendyM2M_VerticalImpact.ipynb

 Notebook for WendyM2M to the target disk perturbed by a vertical satellite impact. 

## Wendy4Daisuke.ipynb

 Notebook from Morgan Bennett for an example of a vertical satellite interaction model. 

## WendyM2M_NonAdiabaticImpact.ipynb

 Notebook for WendyM2M to the target disk constructed from a non-adiabatic impact. 

## WendyM2M.iynb

 Copy from Jo Bovy's simple-m2m/py

## hom2m.py  wendym2m.py

 Copy from Jo Bovy's simple-m2m/py
