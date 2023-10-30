import ACID.ACID as acid
import numpy as np
import glob

# import os

# files = glob.glob('/Users/lucydolan/Documents/GitHub/ACID/tests/data//*/*/*/**.fits') 
# print(files)

# for file in files:
#     os.system('mv "%s" "%s"'%(file, '/Users/lucydolan/Documents/GitHub/ACID/tests/data/'))

def test_run():

    e2ds_files = glob.glob('/Users/lucydolan/Documents/HD189733/July2007/*/*/*/*e2ds*A*.fits')
    s1d_files = glob.glob('/Users/lucydolan/Starbase/problem_frames/*s1d*.fits')
    linelist = '/Users/lucydolan/Starbase/fulllinelist0001.txt'
    save_path = '/Users/lucydolan/Starbase/problem_frames/'

    if len(e2ds_files)==0:
        e2ds_files = glob.glob('./*e2ds*.fits')
        s1d_files = glob.glob('./*s1d*.fits')
        linelist = '/home/lsd/Documents/Starbase/novaprime/Documents/fulllinelist0001.txt'
        save_path = './'

    velocities = np.arange(-25, 25, 0.82)

    # run ACID on e2ds and s1d files
    ACID_results_e2ds = acid.ACID_e2ds(velocities, e2ds_files, linelist, save_path = save_path, order_range = np.arange(41, 45))

    ACID_results_s1d = acid.ACID_e2ds(velocities, s1d_files, linelist, save_path = save_path, order_range = np.arange(41, 45), file_type = 's1d')

# test_run()