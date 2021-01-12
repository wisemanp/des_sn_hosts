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

  real<lower=-2,upper=0> beta; //late time slope of the DTD
  real<lower=0.02,upper=1> tp; // prompt time of the DTD (Gyr)
  real norm; // normalisation of the DTD

}

transformed parameters {
  vector<lower=0>[N] latent_rate; // The model rates
  vector[N] log_latent_rate; //log of the latent rate
  real log_tp; // log of the prompt time
  real<lower=-15,upper=-10> log_norm; // log of the normalisation

  log_tp = log10(tp);
  log_norm = log10(norm);
  for (n in 1:N)
  {
    latent_rate[n] = 1E-18;    //some small number to keep it positive
    for (m in 1:M)
    {
      latent_rate[n]+= phi(age[m],pow(10,log_tp),pow(10,log_norm),beta)*SFH[n][m]; //sum the rate arising from each epoch
    }
  }
  log_latent_rate = log10(latent_rate);
}

model {

  // weakly informative priors

  beta ~ normal(-1,0.5);
  log_tp ~ normal(-1.3,2);
  log_norm ~ normal(-12.7,0.5);

  // likelihood
  lograte_obs ~ normal(log_latent_rate, sigma);

}
