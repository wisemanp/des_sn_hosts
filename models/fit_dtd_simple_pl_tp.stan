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

  real<upper=0> beta; //late time slope of the DTD
  real<lower=0.04> tp; // prompt time of the DTD (Gyr)
  real log_norm; // normalisation of the DTD

}

transformed parameters {
  vector<lower=0>[N] latent_rate; // The model rates
  vector[N] log_latent_rate; //log of the latent rate
  real<lower=0> norm; //

  norm = pow(10,log_norm);
  for (n in 1:N)
  {
    latent_rate[n] = 1E-18;    //some small number to keep it positive
    for (m in 1:M)
    {
      latent_rate[n]+= phi(age[m],tp,norm,beta)*SFH[n][m]; //sum the rate arising from each epoch
    }
    //latent_rate[n] *= 2.3*norm;
  }
  log_latent_rate = log10(latent_rate);
}

model {

  // weakly informative priors

  beta ~ normal(-1,0.3);
  tp ~ cauchy(0.04,0.1);
  log_norm ~ normal(-12.7,0.5);

  // likelihood
  lograte_obs ~ normal(log_latent_rate, sigma);

}
