data {
  int<lower=1> N_lab;
  array[N_lab] real scores_lab;
  array[N_lab] int<lower=0, upper=1> labels;
  int<lower=1> N_unlab;
  array[N_unlab] real scores_unlab;
  real mu;
}
transformed data {
  real delta = 1e-9;
  int<lower=1> N = N_lab + N_unlab;
  array[N] real x;
  for (n1 in 1 : N_lab) {
    x[n1] = scores_lab[n1];
  }
  for (n2 in 1 : N_unlab) {
    x[N_lab + n2] = scores_unlab[n2];
  }
}
parameters {
  real<lower=0> rho;
  real<lower=0> alpha;
  vector[N] eta;
}
transformed parameters {
  vector[N] f;
  {
    matrix[N, N] L_K;
    matrix[N, N] K = gp_exp_quad_cov(x, alpha, rho);
    
    // diagonal elements
    for (n in 1 : N) {
      K[n, n] = K[n, n] + delta; // add small value to diag to ensure positive definiteness
    }
    
    L_K = cholesky_decompose(K);
    // now calculate f. Don't forget to increment by mu
    f = mu + L_K * eta;
  }
}
model {
  rho ~ inv_gamma(3, 0.75);
  alpha ~ normal(0, 1);
  eta ~ normal(0, 1);
  labels ~ bernoulli_logit(f[1 : N_lab]);
}
generated quantities {
  int total_pos;
  total_pos = sum(labels);
  for (n2 in 1 : N_unlab) {
    // increment by the bernoulli_logit_rng function
    total_pos = total_pos + bernoulli_logit_rng(f[N_lab + n2]);
  }
}