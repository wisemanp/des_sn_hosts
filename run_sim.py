import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy import units as u
import time
import itertools
import progressbar
import os
import pickle
import scipy.stats as stats
import scipy.special as sf
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import argparse

from des_mismatch.simulations.sim import Sim,ZPowerSchechterSim
from des_mismatch.functions import features, match
from des_mismatch.utils.utils import Constants

def parser():
    '''FIXME'''
    pass

def main():
    c = Constants()
    sim = ZPowerSchechterSim(Lstar=1*c.Lsun,alpha=-0.25,Lambda=c.lam_D08_z,delta=1.5,r_max=0.2)
    sim.pop_df = sim.synth_pop()
    sim.plot_pop()
if __name__=="__main__":
    main()
