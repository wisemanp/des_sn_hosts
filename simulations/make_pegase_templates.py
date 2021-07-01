from pypegase.pypegase import *
import pandas as pd
import argparse
import multiprocessing
from tqdm import tqdm
import numpy as np

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-z','--z',help='Redshift',default=0.0,type=str)
    parser.add_argument('-ne','--neb',action='store_true')
    args = parser.parse_args()
    return args

def peg_worker(age):
    peg = PEGASE('noneb', ssps=SSP(IMF(IMF.IMF_Kroupa), ejecta=SNII_ejecta.MODEL_B, galactic_winds=True),
                 scenarios=[Scenario(binaries_fraction=0.04, metallicity_ism_0=0, infall=False,
                                     sfr=SFR(SFR.FILE_SFR, p1=1000, p2=1,
                                             filename='/media/data3/wiseman/des/AURA/PEGASE/SFHs/sfh_%i.dat'%age),
                                     metallicity_evolution=True, substellar_fraction=0, neb_emission=True,
                                     extinction=Extinction.NO_EXTINCTION)])
    peg.generate()
    peg.save_to_file('/media/data3/wiseman/des/AURA/templates/%i.dat'%age)


def main(args):
    store = pd.HDFStore('/media/data3/wiseman/des/desdtd/SFHs/SFHs_alt_0.5_Qerf_1.1.h5', 'r')
    ordered_keys = np.sort([int(x.strip('/')) for x in store.keys()])
    pool_size = 10
    results = []
    pool = multiprocessing.Pool(processes=pool_size,maxtasksperchild=1)

    for _ in tqdm (pool.imap_unordered(peg_worker,ordered_keys)):
        results.append(_)
    pool.close()
    pool.join()
    pool.close()
    return results

if __name__=="__main__":
    args = parser()
    main(args)
