
### misc.py

 copy from https://github.com/morganb-phys/VWaves-GaiaDR2

### BB19_RVS-result.fits

 From Gaia Archive, the scipt used is as follows, which is a copy from  https://github.com/morganb-phys/VWaves-GaiaDR2/py/VerticalVelocities.ipynb

SELECT radial_velocity, radial_velocity_error, phot_g_mean_mag, bp_rp,
ra, dec, parallax, parallax_error, pmra, pmra_error, pmdec, pmdec_error
FROM gaiadr2.gaia_source
WHERE radial_velocity IS NOT Null AND parallax_over_error>5.
AND parallax IS NOT Null

