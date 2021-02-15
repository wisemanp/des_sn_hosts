functions {

  real phi(real t, real tpe,real norm){

  //  The delay time distribution

  real dtd = 0 ; // initialise the DTD
  if (t > tpe)
    dtd = norm * pow(t,-1);
  return dtd;
  }



}


data {
  int<lower=0> Nlo; // number of data points
  vector[Nlo] logmass_obs_lo; // x observations
  vector[Nlo] lograte_obs_lo; // y observations
  vector<lower=0>[Nlo] sigma_lo; // heteroskedastic measurement error
  int<lower=0> M; // length of the SFH matrix
  vector[M] age_lo; // age of stellar populations
  matrix[Nlo,M] SFH_lo; // star formation histories

  int<lower=0> Nhi; // number of data points
  vector[Nhi] logmass_obs_hi; // x observations
  vector[Nhi] lograte_obs_hi; // y observations
  vector<lower=0>[Nhi] sigma_hi; // heteroskedastic measurement error
  vector[M] age_hi; // age of stellar populations
  matrix[Nhi,M] SFH_hi; // star formation histories


}

parameters {

  //real<lower=0,upper=1>frac_normhi; // fraction of high-x1 SNe
  real<lower=-2,upper=1.13> log_tpe; // log of the prompt time
  real<lower=log_tpe+0.001,upper=1.14> log_tpl; // log of the late time
}

transformed parameters {
  vector<lower=0>[Nlo] latent_rate_lo; // The model rates
  vector[Nlo] log_latent_rate_lo; //log of the latent rate
  vector<lower=0>[Nhi] latent_rate_hi; // The model rates
  vector[Nhi] log_latent_rate_hi; //log of the latent rate
  real tpe; // log of the prompt time
  real tpl; // log of the late time
  //vector<lower=0,upper=1>[M] frac_prompt; // fraction of SNe that are prompt

  tpe = pow(10,log_tpe);
  tpl = pow(10,log_tpl);


  for (n in 1:Nlo)
  {
    latent_rate_lo[n] = 1E-18;    //some small number to keep it positive
    for (m in 1:M)
    {
      if (age_lo[m] < tpe)
        latent_rate_lo[n]+= 0; //sum the rate arising from each epoch

      else if (age_lo[m]>=tpe)
        if (age_lo[m]< tpl)
          latent_rate_lo[n]+= ((age_lo[m] - tpe)/(tpl - tpe))*phi(age_lo[m],tpe,pow(10,-12.75))*SFH_lo[n][m]; //sum the rate arising from each epoch
        else
          latent_rate_lo[n]+= phi(age_lo[m],tpe,pow(10,-12.75))*SFH_lo[n][m]; //sum the rate arising from each epoch
    }
  }
  log_latent_rate_lo = log10(latent_rate_lo);

  for (n in 1:Nhi)
  {
    latent_rate_hi[n] = 1E-18;    //some small number to keep it positive
    for (m in 1:M)
    {
      if (age_hi[m] < tpe )
        latent_rate_hi[n]+= phi(age_hi[m],0.04,pow(10,-12.75))*SFH_hi[n][m]; //sum the rate arising from each epoch; //sum the rate arising from each epoch
      else if (age_hi[m] >= tpe)
        if (age_hi[m] < tpl)
          latent_rate_hi[n]+= (1-((age_hi[m] - tpe)/(tpl - tpe)))*phi(age_hi[m],0.04,pow(10,-12.75))*SFH_hi[n][m]; //sum the rate arising from each epoch
        else
          latent_rate_hi[n]+= 0;
    }
  }
  log_latent_rate_hi = log10(latent_rate_hi);

}

model {

  // weakly informative priors
  log_tpe ~ normal(-0.3,0.2);
  log_tpl ~ normal(0,0.2);


  // likelihood
  lograte_obs_lo ~ normal(log_latent_rate_lo, sigma_lo);
  lograte_obs_hi ~ normal(log_latent_rate_hi, sigma_hi);

}
