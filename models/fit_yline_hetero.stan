data {

  int<lower=0> N; // number of data points
  vector[N] x_obs; // x observations
  vector[N] y_obs; // y observations
  vector<lower=0>[N] sigma; // heteroskedastic measurement error

  int<lower=0> N_model; // number of data points for line
  vector[N_model] x_model; //where to evaluate the model

}

parameters {

  real slope; //slope of the line
  real intercept; //intercept of the line

}

transformed parameters {

  // latent y values not obscured by measurement error
  vector[N] y_latent = slope * x_obs + intercept;

}

model {

  // weakly informative priors

  slope ~ normal(0,5);
  intercept ~ normal(0,5);

  // likelihood

  y_obs ~ normal(y_latent, sigma);



}

generated quantities {

  vector[N] ppc;
  vector[N_model] line;
  vector[N] log_lik;
  // generate the posterior of the
  // fitted line
  line = slope * x_model + intercept;

  // create posterior samples for PPC
  for (n in 1:N) {

    ppc[n] = normal_rng(slope * x_obs[N] + intercept, sigma[N]);

  // generate the log-likelihood for the model
    for (i in 1:N)
         log_lik[i] = normal_lpdf(y_obs[i] | y_latent, sigma);
  }


}
