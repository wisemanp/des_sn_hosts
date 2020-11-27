functions {

  real phi(real t, real tp, real alpha, real beta){
  /*
    The delay time distribution
  /*
  real dtd = pow((t/tp),alpha) / (pow((t/tp),alpha-beta)+1);
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

  int<lower=0> N_model; // number of data points for line
  vector[N_model] x_model; //where to evaluate the model

}

parameters {

  real<lower=0> alpha; //order of early time polynomial
  real beta; //late time slope of the DTD
  real<lower=0> tp; // prompt time of the DTD (Gyr)
  real<lower=0> log_norm; // normalisation of the DTD

}

transformed parameters {
  vector[N] latent_rate; // The model rates
  real norm; //

  norm = pow(norm,10)
  for n in (1:N){
    for m in (1:M){
      latent_rate[n] += dtd(age[m],tp,alpha,beta)*SFH[n][m])/dtd(1,tp,alpha,beta);
    latent_rate[n] *= 2.3*norm;
    }

  }

  // latent y values not obscured by measurement error
  // vector[N] y_latent = slope * x_obs + intercept;

}

model {

  // weakly informative priors

  alpha ~ normal(13,1);
  beta ~ normal(-1,0.5);
  tp ~ normal(0.2,0.5);
  log_norm ~ normal(-13,1);
  // likelihood

  lograte_obs ~ normal(latent_rate, sigma);



}
