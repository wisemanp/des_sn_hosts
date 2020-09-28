functions {
  real sign(real x_loc) {
    if (x_loc > 0)
      return 1;  
    else
      return 0;
      }
}

data {

  // parameters for the
  // generated data

  int<lower=0> n_obs;
  vector[n_obs] x_obs;
  vector<lower=0>[n_obs] x_err_plus;
  vector<lower=0>[n_obs] x_err_minus;
}

generated quantities {

  // storage for the generated data
  vector[n_obs] x_sim;

  for (n in 1:n_obs) {

    // first pull from the uniform distribution to determine sign of the sim
    x_loc = uniform_rng(0,1);
    x_sign = sign(x_loc);
    // randomly pull x from a normal distribution
    if x_sign > 0:
      x_sim[n] = fabs(normal_rng(x_obs[n], x_err_plus[n]));
    else:
      x_sim[n] = -1*fabs(normal_rng(x_obs[n], x_err_minus[n]));
  }
}
