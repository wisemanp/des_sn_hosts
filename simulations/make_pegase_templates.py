from pypegase.pypegase import *
import pandas as pd
import argparse
import multiprocessing
from tqdm import tqdm
import numpy as np

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-z','--z',help='Redshift',default=0.0,type=str)
    parser.add_argument('-ne','--neb',action='store_true')s
    parser.add_argument('-tr','--time_res',help='Time resolution',default=5,type=int)
    parser.add_argument('-s','--savename',help='Save name',default='new')
    args = parser.parse_args()
    return args

def generate_scenario(age,args):
    s = Scenario(binaries_fraction=0.04, metallicity_ism_0=0, infall=False,
             sfr=SFR(SFR.FILE_SFR, p1=1000, p2=1,
                     filename='/media/data3/wiseman/des/AURA/PEGASE/SFHs/sfh_alt_%i.dat' % age),
             metallicity_evolution=True, substellar_fraction=0, neb_emission=args.neb,
             extinction=Extinction.NO_EXTINCTION)
    return s

def main(args):
    store = pd.HDFStore('/media/data3/wiseman/des/desdtd/SFHs/SFHs_alt_0.5_Qerf_1.1.h5', 'r')
    ordered_keys = np.sort([int(x.strip('/')) for x in store.keys()])
    scenario_list = [generate_scenario(a,args) for a in ordered_keys[::-1][np.arange(0,len(ordered_keys),args.time_res)]]
    peg = PEGASE(args.savename, ssps=SSP(IMF(IMF.IMF_Kroupa), ejecta=SNII_ejecta.MODEL_B, galactic_winds=True),
                 scenarios=scenario_list)
    peg.generate()
    counter=1
    for tf in tqdm(ordered_keys[::-1][np.arange(0,len(ordered_keys),args.time_res)]):

        spec = peg.spectra(scenario=counter)
        counter+=1
        templates = spec.to_pandas()
        templates.to_hdf('/media/data3/wiseman/des/AURA/PEGASE/templates_analytic_%i.h5'%tf, key='main')

    return

if __name__=="__main__":
    args = parser()
    main(args)
