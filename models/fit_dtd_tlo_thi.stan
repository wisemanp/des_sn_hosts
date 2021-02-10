functions {

  real phi(real t, real tpe, real tpl, real norm, real beta){

  //  The delay time distribution

  real dtd = 0 ; // initialise the DTD
  if (t > tpe)
    if (t<=tpl)
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
  real<lower=-20,upper=-5> log_norm; // log of the normalisation of the DTD
  real<lower=-2,upper=1> log_tpe; // log of the prompt time
  real<lower=log_tpe,upper=2> log_tpl; // log of the late time
}

transformed parameters {
  vector<lower=0>[N] latent_rate; // The model rates
  vector[N] log_latent_rate; //log of the latent rate
  real tpe; // log of the prompt time
  real tpl; // log of the late time
  real norm; // normalisation

  tpe = pow(10,log_tpe);
  tpl = pow(10,log_tpl);
  norm = pow(10,log_norm);
  for (n in 1:N)
  {
    latent_rate[n] = 1E-18;    //some small number to keep it positive
    for (m in 1:M)
    {
      latent_rate[n]+= phi(age[m],tpe,tpl,norm,-1)*SFH[n][m]; //sum the rate arising from each epoch
    }
  }
  log_latent_rate = log10(latent_rate);
}

model {

  // weakly informative priors

  //beta ~ normal(-1,0.5);
  log_tpe ~ normal(0,0.2);
  log_tpl ~ normal(1,0.2);
  log_norm ~ normal(-12.7,0.5);

  // likelihood
  lograte_obs ~ normal(log_latent_rate, sigma);

}
