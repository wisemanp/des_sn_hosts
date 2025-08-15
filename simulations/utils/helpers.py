import pandas as pd
import numpy as np
def prep_df_for_BBC(df):
    df = df[df['mB']<25]
    df = df[(df['x1']<3)&(df['x1']>-3)&(df['c']>-0.3)&(df['c']<0.3)&\
                           (df['x1_err']<1)\
                           &(df['c_err']<0.1)    # uncomment to include a colour error cut
                           ]
    df['CID'] = np.arange(len(df),dtype=int)
    #df['CID'] = df['CID'].astype(int)
    df['IDSURVEY'] = 10
    df['TYPE'] = 101
    df.rename(columns={'z':'zHD','mB_err':'mBERR','x1_err':'x1ERR','c_err':'cERR',
                             },inplace=True)
    df = df[df['zHD']>=0.15]
    df['zHDERR'] = 0.0001
    df['zCMB'] = df['zHD']
    df['zCMBERR'] = 0.0001
    df['zHEL'] = df['zHD']
    df['zHELERR'] = 0.0001
    df['VPEC'] =0
    df['VPECERR'] =0
    df['x0']=10**(-0.4*(df['mB']-10.6350))
    df['x0ERR']=0.4*np.log(10)*df['x0'].values*df['mBERR']
    df['COV_x1_x0'] =0
    df['COV_c_x0'] =0
    df['COV_x1_c'] =0
    df['PKMJD'] =56600
    df['PKMJDERR'] =0.1
    df['FITPROB'] =1
    df['PROB_SNNTRAINV19_z_TRAINDES_V19']=1
    df['HOST_LOGMASS'] = np.log10(df['mass'])
    df['HOST_LOG_SFR'] = np.log10(df['sfr'])
    df['HOST_LOG_sSFR'] = np.log10(df['ssfr'])
    df['VARNAMES:'] = 'SN:'
    columns=['VARNAMES:','CID', 'IDSURVEY', 'TYPE', 'mB', 'mBERR', 'cERR', 'x1ERR', 'zHD', 'TYPE', 'zHDERR', 'zCMB',
           'zCMBERR', 'zHEL', 'zHELERR','x0', 'x0ERR','COV_x1_x0', 'COV_c_x0', 'COV_x1_c',
             'VPEC', 'VPECERR', 'PKMJD', 'PKMJDERR', 'FITPROB',
           'PROB_SNNTRAINV19_z_TRAINDES_V19', 'HOST_LOGMASS', 'HOST_LOG_SFR',
           'HOST_LOG_sSFR', 'distmod', 'mass', 'ssfr', 'sfr', 'mean_ages',
           'SN_age', 'rv', 'E', 'host_Av', 'U-R', 'c', 'c_noise', 'c_int', 'x1','x1_int','x1_noise'
           #'prog_age'
            ]
    return df,columns