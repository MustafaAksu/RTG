import numpy as np
cfg = np.load('cfg_fresh/cfg_00002.npz')
dphi = ((cfg['phi'][1]-cfg['phi'][0] + np.pi) % (2*np.pi)) - np.pi
print('Δφ at cfg_00002:', dphi)      # should be very close to 0