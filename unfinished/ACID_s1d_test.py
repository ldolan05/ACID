import numpy as np
import ACID_code as ACID_code
import glob
from astropy.io import fits
import LSD_func_faster as LSD 

# e2ds_files = glob.glob('/Users/lucydolan/Starbase/problem_frames/*e2ds*.fits')
e2ds_files = glob.glob('/Users/lucydolan/Documents/HD189733/July2007/*/*/*/*e2ds*A*.fits')
s1d_files = glob.glob('/Users/lucydolan/Starbase/problem_frames/*s1d*.fits')
linelist = '/Users/lucydolan/Starbase/fulllinelist0001.txt'
save_path = '/Users/lucydolan/Starbase/problem_frames/'

if len(e2ds_files)==0:
    e2ds_files = glob.glob('./*e2ds*.fits')
    s1d_files = glob.glob('./*s1d*.fits')
    linelist = '/home/lsd/Documents/Starbase/novaprime/Documents/fulllinelist0001.txt'
    save_path = './'

# if len(s1d_files) != len(e2ds_files):
#     raise ValueError('Number of s1d and e2ds files do not match')

velocities = np.arange(-25, 25, 0.82)

# run ACID on e2ds and s1d files
ACID_results_e2ds = ACID_code.ACID_e2ds(velocities, e2ds_files, linelist, save_path = save_path, order_range = np.arange(41, 45))
# ACID_results_s1d = ACID.ACID_e2ds(velocities, s1d_files, linelist, save_path = save_path, order_range = np.arange(15, 70), file_type='s1d')

## running on seperate order ranges

# P=2.21857567 #Cegla et al, 2006 - days
# T=2454279.436714 #Cegla et al, 2006

# order_ranges = [np.arange(15, 30), np.arange(31, 50), np.arange(51, 70)]

# order_profs = []
# for order_range in order_ranges:
#     ACID_results_e2ds = ACID.ACID_e2ds(velocities, e2ds_files, linelist, save_path = save_path, order_range = order_range)

#     e2ds_prof_files =  glob.glob('August2007_*_test.fits')
#     for frames in range(len(e2ds_files)):
#         profs = []
#         phase = []
#         for file in e2ds_prof_files:
#             file_o = fits.open(file)
#             profiles = []
#             weights = []
#             for i in range(len(file_o)):
#                 print(i)
#                 print(file_o[i].data[0])
#                 if np.sum(file_o[i].data[0])!=0:
#                     print('adding to profiles for file: %s'%file)
#                     profiles.append(file_o[i].data[0])
#                     weights.append(1/(file_o[i].data[1])**2)
#             profiles = np.array(profiles)
#             profs.append(np.average(profiles,weights = weights, axis = 0))
#             phase.append(((file_o[0].header['BJD']-T)/P)%1)
#     order_profs.append(profs)

# deltavs = []
# for file in e2ds_files:

#     hdu=fits.open('%s'%file)
#     spec=hdu[0].data
#     header=hdu[0].header

#     wave_nonad=LSD.get_wave(spec, header)
#     order_deltavs = []
#     for order in range(len(spec)):
#         print(order)
#         # wave_nonad=hdu[0].header['CRVAL1']+(np.arange(spec.shape[0]))*hdu[0].header['CDELT1']
    
#         brv=np.float128(header['ESO DRS BERV'])

#         wave = wave_nonad*(1.+brv/2.99792458e5)
#         wave = wave[order]
#         wavelengths = np.array(wave, dtype = 'float64')

#         resol1 = (wavelengths[-1]-wavelengths[0])/len(wavelengths)
#         deltav = resol1/(wavelengths[0]+((wavelengths[-1]-wavelengths[0])/2))*2.99792458e5
#         print(deltav)
#         order_deltavs.append(deltav)
#         resol1 = deltav*(wavelengths[0]+((wavelengths[-1]-wavelengths[0])/2))*2.99792458e5
#         # print(resol1)
#     deltavs.append(order_deltavs)

# print('done')

# P=2.21857567 #Cegla et al, 2006 - days
# T=2454279.436714 #Cegla et al, 2006

# deltavs = np.arange(0.8, 0.85, 0.01)

# v_profs = []
# for deltav in deltavs:
#     velocities = np.arange(-25, 25, round(deltav, 2))
#     ACID_results_e2ds = ACID.ACID_e2ds(velocities, e2ds_files, linelist, save_path = save_path, order_range = np.arange(15, 70))

#     e2ds_prof_files =  glob.glob('August2007_*_test.fits')
#     for frames in range(len(e2ds_files)):
#         profs = []
#         phase = []
#         for file in e2ds_prof_files:
#             file_o = fits.open(file)
#             profiles = []
#             weights = []
#             for i in range(len(file_o)):
#                 print(i)
#                 print(file_o[i].data[0])
#                 if np.sum(file_o[i].data[0])!=0:
#                     print('adding to profiles for file: %s'%file)
#                     profiles.append(file_o[i].data[0])
#                     weights.append(1/(file_o[i].data[1])**2)
#             profiles = np.array(profiles)
#             profs.append(np.average(profiles,weights = weights, axis = 0))
#             phase.append(((file_o[0].header['BJD']-T)/P)%1)
#     v_profs.append(profs)


## temperature

# P=2.21857567 #Cegla et al, 2006 - days
# T=2454279.436714 #Cegla et al, 2006

# temp = []
# phase = []
# for file in e2ds_files:

#     hdu=fits.open('%s'%file)
#     spec=hdu[0].data
#     header=hdu[0].header

#     phase.append(((header['ESO DRS BJD']-T)/P)%1)
#     temp.append(header['ESO DRS CAL TH RMS ORDER40'])

# e2ds_prof_files = glob.glob('%s/August2007**.fits'%save_path)

# profs = []
# phase = []
# for file in e2ds_prof_files:
#     file_o = fits.open(file)
#     profiles = []
#     weights = []
#     for i in range(len(file_o)):
#         print(i)
#         print(file_o[i].data[0])
#         if np.sum(file_o[i].data[0])!=0:
#             print('adding to profiles for file: %s'%file)
#             profiles.append(file_o[i].data[0])
#             weights.append(1/(file_o[i].data[1])**2)
#     profiles = np.array(profiles)
#     profs.append(np.average(profiles,weights = weights, axis = 0))
#     phase.append(((file_o[0].header['BJD']-T)/P)%1)

print('done')