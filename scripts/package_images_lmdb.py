from glob import glob
import sys
sys.path = ['..'] + sys.path
import elytra.sys_utils as sys_utils
import pdb; pdb.set_trace() 

db_path = 'interhand.lmdb'
fnames = glob('../data/InterHand/images/val/*/*/*/*')

import random
random.shuffle(fnames)
fnames = fnames[:1000]

map_size = len(fnames) * 5130240
keys = [fname.replace('../data/InterHand/images/', '') for fname in fnames]
sys_utils.package_lmdb(db_path, map_size, fnames, keys)
