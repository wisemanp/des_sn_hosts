
data {

  // example parameters for the
  // generated data

  int<lower=0> n_obs;
  vector[n_obs] x_obs;
  vector<lower=0>[n_obs] x_err;
}

generated quantities {

  // storage for the generated data
  vector[n_obs] x_sim;

  for (n in 1:n_obs) {
    // randomly pull x from a normal distribution
    x_sim[n] = normal_rng(x_obs[n], x_err[n]);
  }
}
