functions {
  int sign(real x_loc) {
    if (x_loc > 0.5)
      return 1;
    else
      return 0;
      }

  real norm_sign(real norm_out) {
    if (norm_out > 0)
      return norm_out;
    else
      return -1*norm_out;
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
  vector[n_obs] x_loc;
  vector[n_obs] x_sign;
  for (n in 1:n_obs) {

    // first pull from the uniform distribution to determine sign of the sim
    x_loc[n] = uniform_rng(0,1);
    x_sign[n] = sign(x_loc[n]);
    // randomly pull x from a normal distribution

    if (x_sign[n] > 0)
      x_sim[n] = x_obs[n] + norm_sign(fabs(normal_rng(0, x_err_plus[n])));
    else
      x_sim[n] = x_obs[n] - norm_sign(fabs(normal_rng(0, x_err_minus[n])));
  }
}
