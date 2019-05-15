"""
Collect the various splitting strategies in one place
"""

import numpy as np
from scipy.ndimage import generic_filter
from scipy.spatial.distance import cdist
from numpy.random import rand
import tools
from functools import reduce


def gensig_model(X, minlength=1, maxlength=None, lam=0.0):
    N,D = X.shape
    over_sqrtD = 1./np.sqrt(D)
    cs = np.cumsum(X,0)

    def sigma(a,b):
        length = (b-a)
        if minlength:
            if length < minlength: return np.inf
        if maxlength:
            if length > maxlength: return np.inf

        tot = cs[b-1].copy()
        if a > 0:
            tot -= cs[a-1]
        signs = np.sign(tot)
        return -over_sqrtD*(signs*tot).sum()
    return sigma


####################
# Greedy
####################


def greedysplit(n, k, sigma):
    """ Do a greedy split """
    splits = [n]
    s = sigma(0,n)

    def score(splits, sigma):
        splits = sorted(splits)
        return sum( sigma(a,b) for (a,b) in tools.seg_iter(splits) )

    while k > 0:
        usedinds = set(splits)
        new = min( ( score( splits + [i], sigma), splits + [i] )
                for i in range(1,n) if i not in usedinds )
        splits = new[1]
        s = new[0]
        k -= 1
    return sorted(splits), s

def bestsplit(low, high, sigma, minlength=1, maxlength=None):
    """ Find the best split inside of a region """
    length = high-low
    if length < 2*minlength:
        return (np.inf, np.inf, low)
    best = min( ((sigma(low,j), sigma(j, high), j) for j in range(low+1,high)), key=lambda x: x[0]+x[1] )
    return best

def refine(splits, sigma, n=1):
    """ Given some splits, refine them a step """
    oldsplits = splits[:]
    counter = 0
    n = n or np.inf

    while counter < n:
        splits = [0]+splits
        n = len(splits) - 2
        new = [splits[0]]
        for i in range(n):
            out = bestsplit(splits[i], splits[i+2], sigma)
            new.append(out[2])
        new.append(splits[-1])
        splits = new[1:]

        if splits == oldsplits:
            break
        oldsplits = splits[:]
        counter += 1

    return splits