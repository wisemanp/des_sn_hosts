import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('dark_background')
import sys
from tqdm import tqdm

class Completeness():
    '''A class to calculate completeness of a galaxy survey following Johnston Teodoro & Hendry 2011'''

    def __init__(self,df,dZ=0.1,dM=0.1,band='g',mlim_B = 15):
        self.df = df
        self.dZ = dZ
        self.dM = dM
        self.b = band
        self.mlim_B = mlim_B


    def r_i(self,row):
        Z = row['mu']
        M = row['M_%s'%self.b]
        #print(M,Z)
        Mlim_B = self.mlim_B - (Z-self.dZ)
        sel = self.test_df[(self.test_df['mu']<Z)&(self.test_df['mu']>Z-self.dZ)&(self.test_df['M_%s'%self.b]<M)&(self.test_df['M_%s'%self.b]>Mlim_B)]
        #print('ri',len(sel))
        return len(sel)

    def n_i(self,row,mlim_star):
        Z = row['mu']
        Mlim_star = mlim_star - Z
        Mlim_B = self.mlim_B - (Z-self.dZ)
        M = row['M_%s'%self.b]
        #if M>Mlim_star:
        #    sel = self.test_df[(self.test_df['mu']<Z)&(self.test_df['mu']>Z-self.dZ)&(self.test_df['M_%s'%self.b]<M)&(self.test_df['M_%s'%self.b]>Mlim_B)]
        #print('%.2f < M <%.2f'%(Mlim_B,Mlim_star))
        #else:
        sel = self.test_df[(self.test_df['mu']<Z)&(self.test_df['mu']>(Z-self.dZ))&(self.test_df['M_%s'%self.b]<Mlim_star)&(self.test_df['M_%s'%self.b]>Mlim_B)]
        #print(len(sel))
        #print('ni',len(sel))
        return len(sel)

    def q_i(self,row):
        Z = row['mu']
        M = row['M_%s'%self.b]

        sel = self.df[(self.df['mu']<Z)&(self.df['M_%s'%self.b]>(M-self.dM))&(self.df['M_%s'%self.b]<M)]
        return len(sel)

    def t_i(self,row,mlim_star):
        Z = row['mu']
        M = row['M_%s'%self.b]
        Mlim_B = self.mlim_B - Z
        sel = self.df[(self.df['MAG_AUTO_%s'%self.b.capitalize()]<mlim_star)&(self.df['M_%s'%self.b]>(M-self.dM))&(self.df['M_%s'%self.b]<M)]
        return len(sel)

    def var_zeta_i(self,ni):
        #print((ni -1),(ni+1),(ni - 1)/(ni + 1))

        return (1/12)*(ni - 1)/(ni + 1)

    def zeta_i(self,row,mlim_star):

        r_i = self.r_i(row)
        n_i = self.n_i(row,mlim_star)

        zeta = r_i/(n_i+1)
        var = self.var_zeta_i(n_i)
        #print(r_i,n_i,zeta,var)
        row['z_i'] = zeta
        row['var_z_i'] = var
        return row



    def tau_i(self,row,mlim_star):
        t_i = self.t_i(row,mlim_star)
        tau = self.q_i(row)/(t_i+1)
        var = self.var_tau_i(t_i)
        row['t_i'] = tau
        row['var_t_i'] = var
        return row

    def var_tau_i(self,ti):
        return (1/12)*(ti - 1)/(ti + 1)

    def Tc(self,mlim_star):
        self.test_df = self.df[self.df['MAG_AUTO_%s'%self.b.capitalize()]<mlim_star]
        self.test_df['z_i'] = ''
        self.test_df['var_z_i'] = ''
        self.test_df = self.test_df.apply(lambda x: self.zeta_i(x,mlim_star),axis=1)
        #print('zeta i',self.df['z_i'].values - 0.5)
        print(mlim_star)
        #print('zeta i',self.df['z_i'].values )
        #print('mean of zeta i',self.test_df['z_i'].mean())

        #print('var zeta i',self.test_df['var_z_i'].values)
        #print(np.sum(self.test_df['z_i'].fillna(0).values - 0.5))
        self.test_df['var_z_i'].fillna(1/12,inplace=True)
        good_var = np.abs(self.test_df['var_z_i'].values)

        #print(len(good_var),len(good_var[good_var>0]))
        #print(np.sum(good_var**0.5))
        Tci = np.sum(self.test_df['z_i'].fillna(0).values - 0.5)/np.sum(good_var**0.5)
        print('Tc_i',Tci)
        return Tci

    def Tv(self,mlim_star):
        self.df['t_i'] = ''
        self.df['var_t_i'] = ''
        self.df = self.df.apply(lambda x: self.tau_i(x,mlim_star),axis=1)
        sigma_t_i = np.sum(self.df['t_i'] - 0.5)
        sigma_var_i = np.sum(self.df['var_t_i'])
        return sigma_t_i/(sigma_var_i**0.5)


def estimate_Tc(C, mlim_stars):
    Tcs = []
    for m in tqdm(mlim_stars):
        Tcs.append(C.Tc(m))
    return Tcs

def main():
    if not sys.argv[1]:
        fname = '/media/data3/wiseman/des/coadding/results/deep/deep_X3_hostlib.h5'
    else:
        fname = sys.argv[1]

    hostlib = pd.read_hdf(fname,key='main')

    C = Completeness(hostlib.sample(50000),dZ=0.1,dM=0.1)
    mlim_stars = np.linspace(23,30,36)
    Tcs = estimate_Tc(C,mlim_stars)
    f,ax=plt.subplots()
    ax.scatter(mlim_stars,Tcs,label='$T_C$')
    #ax.scatter(mlim_stars,Tvs,label='$T_V$')
    ax.legend()
    plt.savefig('/media/data3/wiseman/des/desdtd/figs/completeness')
    arr = np.zeros((len(mlim_stars),2))
    arr[:,0] = mlim_stars
    arr[:,1] = Tcs

    np.savetxt('/media/data3/wiseman/des/desdtd/completeness/SN-X3_50k_completeness.dat',arr)
if __name__=="__main__":
    main()
