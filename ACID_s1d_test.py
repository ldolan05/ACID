import numpy as np
import ACID as ACID
import glob

e2ds_files = glob.glob('/Users/lucydolan/Starbase/problem_frames/*e2ds*.fits')
s1d_files = glob.glob('/Users/lucydolan/Starbase/problem_frames/*s1d*.fits')

if len(s1d_files) != len(e2ds_files):
    raise ValueError('Number of s1d and e2ds files do not match')

