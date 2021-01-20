functions {

  real phi(real t, real tp, real norm, real beta){

  //  The delay time distribution

  real dtd = 0 ; // initialise the DTD
  if (t > tp)
    dtd = norm * pow(t,beta);
  return dtd;
  }



}


data {

  int<lower=0> N; // number of data points
  vector[N] logmass_obs; // x observations
  vector[N] lograte_obs; // y observations
  vector<lower=0>[N] sigma; // heteroskedastic measurement error
  int<lower=0> M; // length of the SFH matrix
  vector[M] age; // age of stellar populations
  matrix[N,M] SFH; // star formation histories

}

parameters {
  real<upper=0> log_tp; // prompt time of the DTD (Gyr)
  }

transformed parameters {
  vector<lower=0>[N] latent_rate; // The model rates
  vector[N] log_latent_rate; //log of the latent rate
  real<lower=0> tp;

  tp = pow(10,log_tp);
  for (n in 1:N)
  {
    latent_rate[n] = 1E-18;    //some small number to keep it positive
    for (m in 1:M)
    {
      latent_rate[n]+= phi(age[m],tp,1.75E-13,-1)*SFH[n][m]; //sum the rate arising from each epoch
    }
    //latent_rate[n] *= 2.3*norm;
  }
  log_latent_rate = log10(latent_rate);
}

model {

  // weakly informative priors

  log_tp ~ normal(-1.3,0.2);


  // likelihood
  lograte_obs ~ normal(log_latent_rate, sigma);

}
