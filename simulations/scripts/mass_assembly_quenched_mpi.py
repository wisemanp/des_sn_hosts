import numpy as np
import pandas as pd
from astropy.cosmology import z_at_value
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(70,0.3)
import os
from scipy.special import erf
from astropy import units as u
import numpy as np
from tqdm import tqdm
import argparse
import yaml
from des_sn_hosts.utils.utils import MyPool


def phi_t(t,tp,alpha,s):
    '''Functional form of the delay time distribution'''
    return (t/tp)**alpha / ((t/tp)**(alpha-s)+1)


def psi_Mz(M,z):
    #print("Nominal SFRs: \n",2.00*np.exp(1.33*z)*(M/1E+10)**0.7)
    return 2.00*np.exp(1.33*z)*(M/1E+10)**0.7

def logMQ_z(z):
    Mlo,zlo = 10.43,0.9
    Mhi,zhi = 8.56,5.6
    #print ("Quenching masses: \n",((Mlo+zlo*np.log10(1+z))*(z<=1.5)) + ((Mhi+zhi*np.log10(1+z))*(z>1.5)))
    return ((Mlo+zlo*np.log10(1+z))*(z<=1.5)) + ((Mhi+zhi*np.log10(1+z))*(z>1.5))

def pQ_Mz(M,z):
    #print("Quenching function: \n",0.5*(1-erf((np.log10(M)-logMQ_z(z))/1.5)))
    return 0.5*(1-erf((np.log10(M)-logMQ_z(z))/0.5))

def sfr_Mz(M,z):
    return pQ_Mz(M,z) * psi_Mz(M,z)


def psi_Mz_alt(M,z):
    #print("Alternate SFRs: \n",36.4*(M/1E+10)**0.7 * np.exp(1.9*z)/(np.exp(1.7*z)+np.exp(0.2*z)))
    #return 36.4*(M/1E+10)**0.7 * np.exp(1.9*z)/(np.exp(1.7*z)+np.exp(0.2*z))
    return ((np.array(M)/1E+10)**0.7) * np.exp(1.9*(z))/(np.exp(1.7*(z-2))+np.exp(0.2*(z-2)))

def logMQ_z_alt_init(z):

    #print ("Quenching masses: \n",10.077 + 0.636*z)
    return (13.077 + 0.636*z) * (z<=2) + (13.077 + 0.636*2) * (z>2)


def pQ_Mz_alt_init(M,z):

    #print("Quenching function: \n",0.5*(1-erf((np.log10(M)-logMQ_z_alt(z))/1.5)))
    return 0.5*(1-erf((np.log10(M)-logMQ_z_alt_init(z))/1.1))

def logMQ_z_alt(z):

    #print ("Quenching masses: \n",10.077 + 0.636*z)
    return (10.077 + 0.636*z) * (z<=2) + (10.077 + 0.636*2) * (z>2)
def pQ_Mz_alt(M,z,Mq):

    #print("Quenching function: \n",0.5*(1-erf((np.log10(M)-logMQ_z_alt(z))/1.5)))
    return (1-erf((np.log10(M)-(np.log10(Mq)-0.85))/0.5)) #0.5*

def draw_pQ(M,z):
    p_arr = [False,True]

    pq = pQ_Mz_alt_init(M,z)

    isq = np.random.choice(p_arr,p=[pq,1-pq])
    #print(z,np.log10(M),pq,isq)
    return isq


def pmin_z(z):
    return 1-((z-10)/10)**2

def pQ_Mz_ft(M,z,isq,mqs):
    pq = []
    isqs = []
    isq=np.array(isq)
    #print(len(isq[isq==True]))
    for counter,m in enumerate(M):

        if isq[counter]:
            penalty =pQ_Mz_alt(m,z,mqs[counter]) #pmin_z(z) + (1 - pmin_z(z))*
            pq.append(penalty)
            #print('Galaxy %i of mass %.2f is quenching, doing this penalty: %.2f at this mass: %.2f'%(counter,np.log10(m),penalty, np.log10(mqs[counter])))
            isqs.append(True)
        else:
            new_isq = draw_pQ(m,z)
            if new_isq:
                pq.append(pQ_Mz_alt(m,z,m))
                isqs.append(True)
                mqs[counter] = m
            else:
                pq.append(1)
                isqs.append(False)
    #print(isqs,pq)
    return isqs,pq,mqs


def pQ_Mz_ft2(M,z,isq):
    pq = []
    isqs = []
    isq=np.array(isq)
    #print(len(isq[isq==True]))
    for counter,m in enumerate(M):

        if isq[counter]:
            pq.append(pmin_z(z))
            isqs.append(True)
        else:
            new_isq = draw_pQ(m,z)
            if new_isq:
                pq.append(pmin_z(z))
                isqs.append(True)
            else:
                pq.append(1)
                isqs.append(False)
    #print(isqs,pq)
    return isqs,pq



def fml_t(t):
    return 0.046*np.log((t/0.276)+1)

def sfr_Mz_alt(M,z,isq,mqs):
    isqs,pqs,mqs = pQ_Mz_ft(M,z,isq,mqs)
    return pqs * psi_Mz_alt(M,z), isqs,mqs


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dt','--dt',help='Time step (Myr)',default=0.5,type=float)
    parser.add_argument('-es','--early_step',help='Early Universe T_F step (Myr)',default=25,type=float)
    parser.add_argument('-ls','--late_step',help='Late Universe T_F step (Myr)',default=50, type=float)
    parser.add_argument('-ts','--tstart',help='Epoch to start the tracks (working backwards) (yr)',default = 0,type=float)
    parser.add_argument('-c','--config',help='Config filename',default='/home/wiseman/code/des_sn_hosts/config/config_rates.yaml')
    parser.add_argument('-n','--n',help='Number of iterations of each galaxy',default=100,type=float)
    args = parser.parse_args()
    return args

def script_worker(worker_args):
    args,tf,save_dir = [worker_args[i] for i in range(len(worker_args))]
    dt = args.dt
    N=args.n

    ages = np.arange(0,tf+dt,dt)
    m = [1E+6 for _ in range(N)]
    m_formed = [ [] for _ in range(N) ]
    m_lost_tot = [[0] for _ in range(N)]
    m_arr = [ [] for _ in range(N) ]
    is_quenched = [False for _ in range(N)]
    ts = []
    zs = []
    mqs = [0 for _ in range(N)]
    for counter,age in enumerate(tqdm(ages)):
        t = tf-age
        ts.append(t)
        #print("Current epoch: %.f Myr"%t, 'Age: ',age)
        try:
            z_t = z_at_value(cosmo.lookback_time,t*u.Myr,zmin=0)
        except:
            z_t = 0
        zs.append(z_t)
        #print("current redshift: %.2f"%z_t)
        m_created, is_quenched,mqs = sfr_Mz_alt(m,z_t,is_quenched,mqs)
        m_created = np.array(m_created)*dt*1E+6
        [m_formed[n].append(m_created[n]) for n in range(N)]
        #print("Mass formed in the last %3f Myr: %2g Msun"%(dt,m_created))
        #print("Mass formed at each epoch so far: ",m_formed)
        taus = ages[:counter+1][::-1]
        #print("Time since epochs of star formation: ",taus)
        f_ml= fml_t(taus)
        #print("Fractional mass lost since each epoch",f_ml)
        ml = f_ml * np.array([m_formed[n] for n in range(N)])
        #m_lost_tot = np.concatenate([m_lost_tot,[0]])

        if counter>1:
            #print('trying to subtract m_lost_tot',m_lost_tot,'from ml ',ml)
            new_ml = [np.sum(ml[n][:counter]- m_lost_tot[n]) for n in range(N)]
        else:
            new_ml = [np.sum(ml[n]) for n in range(N)]
        #print("New mass loss this cycle",new_ml)
        m_lost_tot = [ml[n] for n in range(N)]
        #print("Current array of masses lost",[ "{:0.2e}".format(x) for x in m_lost_tot ])
        #ml_tot = ml - m_lost
        #m_lost = ml_tot
        m = [m[n] + m_created[n] - new_ml[n] for n in range(N)]
        #print("Final mass of this epoch: %.1e"%m)
        #print("#"*100)
        [m_arr[n].append(m[n]) for n in range(N)]
    #print (m_formed)
    #print(m_lost_tot)

    final_age_weights = [m_formed[n] - m_lost_tot[n] for n in range(N)]

    #print([np.log10(m_arr[n][-1]) for n in range(N)])

    print("Saving to: ",os.path.join(save_dir,'SFHs_alt_%.1f_Qerf_1.1_newQmass_test.h5'%dt))
    df = pd.DataFrame()
    #print(track)

    for n in range(N):

        track = np.array([ts,zs,ages,m_formed[n],final_age_weights[n],m_arr[n]]).T
        new_df = pd.DataFrame(track,columns=['t','z','age','m_formed','final_age_weights','m_tot'],index=[n]*len(ages))
        
        df = df.append(new_df)
    df.to_hdf(os.path.join(save_dir,'SFHs_alt_%.1f_quenched.h5'%dt),key='%3.0f'%tf)

def main(args):
    config=yaml.load(open(args.config))
    save_dir = config['rates_root']+'SFHs/'
    dt = args.dt
    N=args.n
    if args.tstart ==0:
        tfs = np.concatenate([np.arange(1000,10000,args.late_step),np.arange(10000,13000,args.early_step)])
    elif args.tstart <10000:
        tfs = np.concatenate([np.arange(args.tstart,10000,args.late_step),np.arange(10000,13000,args.early_step)])
    else:
        tfs = np.arange(args.tstart,13000,args.early_step)

    worker_args = [[args,tf,save_dir] for tf in tfs]
    pool_size = 16
    pool = MyPool(processes=pool_size)
    for _ in tqdm(pool.imap_unordered(script_worker,worker_args),total=len(worker_args)):
        pass
    pool.close()
    pool.join()

if __name__=="__main__":
    main(parser())
