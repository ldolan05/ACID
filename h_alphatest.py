import glob
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import ACID
from specutils.analysis import equivalent_width as eww
from specutils import Spectrum1D
import astropy.units as u


T = 2454279.436714
P = 2.21857567

# line = 'H-alpha'
line = 'Ca H'

if line == 'H-alpha':
    wave_min = 6562.81-2
    wave_max = 6562.81+2
    order = 67

if line == 'CaHK':
    wave_min = 3933.664-2
    wave_max = 3968.470+2
    order = 6

if line == 'Ca K':
    wave_min = 3933.664-2
    wave_max = 3933.664+2
    order = 6

if line == 'Ca H':
    wave_min = 3968.470-2
    wave_max = 3968.470+2
    order = 6

folder_name=("/Users/lucydolan/Starbase/HD189733/July2007/*/*/*/")
e2ds_files=glob.glob("{0}*e2ds_A*.fits".format(folder_name))
print(e2ds_files)

frame_wavelengths, frames, frame_errors, sns, telluric_spec = ACID.read_in_frames(order, e2ds_files)

frame_wavelengths = np.array(frame_wavelengths)
for f in range(len(frames)):
    frames[f] = frames[f]/np.median(frames[f])
frames = np.array(frames)

plt.figure()
plt.ylabel('%s Equivalent Width'%line)
plt.xlabel('Phase')
phases = np.zeros((len(frames), ))
for file_no in range(len(e2ds_files)):
    spec=frames[file_no]
    wave=frame_wavelengths[file_no]
    hdu = fits.open(e2ds_files[file_no])
    phase = ((hdu[0].header['HIERARCH ESO DRS BJD']-T)/P)%1
    phases[file_no] = phase-np.round(phase)
    idx = np.logical_and(wave<wave_max, wave>wave_min)
    # plt.figure()
    plt.plot(wave[idx], spec[idx])
    # plt.show()
    spectrum = Spectrum1D(spectral_axis = wave[idx]*u.AA, flux = spec[idx]*u.Unit('photon AA-1'))
    # ew = eww(spectrum)
    # plt.scatter(phases[file_no], ew, color = 'k')

idx_sort = phases.argsort()
phases = phases[idx_sort]
frames = frames[idx_sort]
frame_wavelengths = frame_wavelengths[idx_sort]
sns =np.array(sns)
sns =sns[idx_sort]

# idx_wave = np.logical_and(frame_wavelengths<6562.81+4, frame_wavelengths>6562.81-4)
# frames = frames[idx_wave]
# frame_wavelengths = frame_wavelengths[idx_wave]

plt.figure()
plt.title('%s - Median frame (0.02<phase<0.03) - frame'%line)
idx = (phases>0.02)
residual = np.median(frames[idx, :][phases[idx]<0.03], axis = 0)-frames
residual[residual<-4] = 0.
plt.pcolormesh(frame_wavelengths, phases, residual)
plt.colorbar(label = 'Residual Flux')
plt.xlabel('Wavelength (Angstroms)')
plt.ylabel('Phase')
plt.xlim(wave_min, wave_max)
plt.show()

# plt.figure()
# plt.plot(phases, sns)
# plt.xlabel('Phase')
# plt.ylabel('S/N')
# plt.show()