data {

  int<lower=0> N; // number of data points
  vector[N] x_obs; // x observations
  vector[N] y_obs; // y observations
  vector<lower=0>[N] sigma; // heteroskedastic measurement error
  int<lower=0> N_model; // number of data points for line
  vector[N_model] x_model; //where to evaluate the model

}

parameters {

  real slope; // slope of the line
  real intercept; // intercept of the line
  real<lower=0> dispersion; // intrinsic dispersion of the data
}

transformed parameters {

  // latent y values not obscured by measurement error
  vector[N] y_latent = slope * x_obs + intercept;
  vector[N] sigma_tot = sqrt(pow(sigma,2) + pow(dispersion,2));
}

}

model {

  // weakly informative priors

  slope ~ normal(0,5);
  intercept ~ normal(0,5);
  dispersion ~ cauchy(0,5);

  // likelihood

  y_obs ~ normal(y_latent, sigma_tot);



}

generated quantities {

  vector[N] ppc;
  vector[N_model] line;
  vector[N] log_lik;
  real log_lik_sum;
  // generate the posterior of the
  // fitted line
  line = slope * x_model + intercept;

  // create posterior samples for PPC
  for (n in 1:N) {

    ppc[n] = normal_rng(slope * x_obs[N] + intercept, sigma[N]);

  // generate the log-likelihood for the model
    for (i in 1:N)
         log_lik[i] = normal_lpdf(y_obs[i] | y_latent, sigma);
    log_lik_sum = sum(log_lik);
  }


}
