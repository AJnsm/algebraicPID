"""
Template code to download interesting neuromaps data, calculate PID measures
of interest and run a spin statistical test. The script implements two
different set-ups, with different number of sources:

    * Set-up 1: the target variable is the Margulies principal functional
      gradient, and the two sources are the myelin and cortical thickness maps.

    * Set-up 2: the target is the same as above, and the five sources are MEG
      power in the five common frequency bands (delta, theta, alpha, beta,
      gamma).

Pedro Mediano, Abel Jansma, Fernando Rosas
May 2024
"""

import neuromaps
from neuromaps.datasets import fetch_annotation
from neuromaps.nulls import alexander_bloch
from neuromaps.transforms import fslr_to_fslr, mni152_to_fslr
from neuromaps.images import load_data

import pickle
import numpy as np
from numpy.random import randn

def pid_magic(X, y):
    """
    Calculate any desirable PID atom.

    Parameters
    ----------
    X : iter of np.ndarray
        Iterable with one 1D np.ndarray for each source variable. Each
        np.ndarray must be of the same length as y.

    y : np.ndarray
        Target variable.

    Returns
    -------
    val : float
        The value of your favourite PID atom.
    """
    assert(all([len(x) == len(y) for x in X]), \
            'Number of samples in X and y does not match')

    ## FIXME: Implement PID magic here
    
    return randn()


## Set-up 1: two sources
# Download data
fc_grad = fetch_annotation(source='margulies2016', desc='fcgradient01', space='fsLR', den='32k')
myelin  = fetch_annotation(source='hcps1200', desc='myelinmap', space='fsLR', den='32k')
thick   = fetch_annotation(source='hcps1200', desc='thickness', space='fsLR', den='32k')

sources = [load_data(myelin), load_data(thick)]

# Calculate PID for the real data
real_pid = pid_magic(sources, load_data(fc_grad))

# Calculate PID for each sample from the null distribution
n_perm = 100
rotated = alexander_bloch(fc_grad, atlas='fslr', density='32k', n_perm=n_perm, seed=1234)
null_pid = np.array([pid_magic(sources, rotated[:,i]) for i in range(n_perm)])

print(f'p-value = {np.mean(real_pid < null_pid)}')


## Set-up 2: five sources
# Download data -- this time we have to downsample the target to a lower
# density to match the source data
fc_grad = fetch_annotation(source='margulies2016', desc='fcgradient01', space='fsLR', den='32k')
fc_lowres = fslr_to_fslr(fc_grad, target_density='4k')

delta  = fetch_annotation(source='hcps1200', desc='megdelta', space='fsLR', den='4k')
theta  = fetch_annotation(source='hcps1200', desc='megtheta', space='fsLR', den='4k')
alpha  = fetch_annotation(source='hcps1200', desc='megalpha', space='fsLR', den='4k')
beta   = fetch_annotation(source='hcps1200', desc='megbeta', space='fsLR', den='4k')
gamma1 = fetch_annotation(source='hcps1200', desc='meggamma1', space='fsLR', den='4k')
gamma2 = fetch_annotation(source='hcps1200', desc='meggamma2', space='fsLR', den='4k')

sources = [load_data(delta), load_data(theta), load_data(alpha), \
           load_data(beta), load_data(gamma1) + load_data(gamma2)]

# Calculate PID for each sample from the null distribution
n_perm = 100
rotated = alexander_bloch(fc_lowres, atlas='fslr', density='4k', n_perm=n_perm, seed=4321)
null_pid = np.array([pid_magic(sources, rotated[:,i]) for i in range(n_perm)])

print(f'p-value = {np.mean(real_pid < null_pid)}')


## Bonus: Saving data and alternative target
with open('fc_lowres.pkl', 'wb') as f:
    pickle.dump(fc_lowres, f)

neurosynth_mni = fetch_annotation(source='neurosynth', desc='cogpc1', space='MNI152', den='2mm')
neurosynth_32 = mni152_to_fslr(neurosynth_mni, '32k')
neurosynth_4 = fslr_to_fslr(neurosynth_32, target_density='4k')

with open('neurosynth_32.pkl', 'wb') as f:
    pickle.dump(neurosynth_32, f)

with open('neurosynth_4.pkl', 'wb') as f:
    pickle.dump(neurosynth_4, f)

# These files can be later read with pickle as usual
with open('fc_lowres.pkl', 'rb') as f:
    fc_lowres = pickle.load(f)

# You should be able to run e.g. print(load_data(fc_lowres))

