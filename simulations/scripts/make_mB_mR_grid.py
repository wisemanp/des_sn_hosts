import sncosmo
import numpy as np
from tqdm import tqdm
from astropy.cosmology import FlatLambdaCDM
cosmo =FlatLambdaCDM(H0=70,Om0=0.3)
t0=57300
r_mags = {}
for z in tqdm(np.arange(0.15,0.775,0.025)):
    print('z: ',z)
    r_mags[z] = {}
    for x1 in np.arange(-3,3.2,0.2):
        print('x1: ',x1)
        r_mags[z][x1]={}
        for c in np.arange(-0.3,0.32,0.02):
            r_mags[z][x1][c]={}
            for MB in np.arange(-20,-17.9,0.1):
                
                mB = cosmo.distmod(z).value+MB -(0.15*x1) +(3.1*c)
                x0=10**(-0.4*(mB-10.635))
                model = sncosmo.Model(source='salt2')
                model.set(z=z,
                            t0=57300,
                            x0=x0,
                            c=c,
                            x1=x1)
                rmag = model.source_peakmag(band='sdssr',magsys='ab')
                gmag = model.source_peakmag(band='sdssg',magsys='ab')
                imag = model.source_peakmag(band='sdssi',magsys='ab')
                r_mags[z][x1][c][MB] = {'mag_r':rmag,'mag_i':imag,'g-r':gmag-rmag}
import pickle
with open('/media/data3/wiseman/des/desdtd/efficiencies/SALT2_mB_obsmags_map.pkl','wb') as f:
    pickle.dump(r_mags,f)
    f.close()
