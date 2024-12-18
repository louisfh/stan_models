data {
    int<lower=0> N_unlabeled;
    int<lower=0> N_labeled;
    vector[N_unlabeled] scores_unlabeled;
    vector[N_labeled] scores_labeled;
    array[N_labeled] int<lower=0, upper=1> labels;
    vector[N_labeled] sampling_weight; // p(sampling). Not used in GP model
}
transformed data {
  //concat labeled and unlabeled scores //
  array[N_labeled + N_unlabeled] real all_scores; 
  for (s_id in 1 : N_labeled) {
    all_scores[s_id] = scores_labeled[s_id];
  }
  for (s_id in 1 : N_unlabeled) {
    all_scores[N_labeled + s_id] = scores_unlabeled[s_id];
  }
  real<lower=0> sd_scores = sd(all_scores);

  //////////////////////////////////
  /// make points at which latent GP is evaluated ///
  int N_points = 150; // Number of inducing points
  // try spacing them so they're evenly spaces in the density of the data
  // i.e. order the data, find step size, and then space the inducing points evenly

  real step_size = (max(scores_unlabeled) - min(scores_unlabeled)) / N_points;
  array[N_points] real eval_points;
  for (n in 1 : N_points) {
    eval_points[n] = min(all_scores) + n * step_size;
  }
  int xlen = N_points + N_labeled;
  array[xlen] real x; // all the points at which the GP will be evaluated
  for (n in 1 : N_labeled) {
    x[n] = scores_labeled[n]; // labeled points
  }
  for (n in 1 : N_points) {
    x[N_labeled + n] = eval_points[n]; // inducing points
  }
  array[N_points] int num_clips = rep_array(0, N_points); 
  // loop through all the unlabeled clips and assign them to the nearest inducing point //
  for (clip in 1 : N_unlabeled) {
    real min_dist = 1000; // initialize to a large value
    int nearest_point_idx;
    for (point in 1 : N_points) {
      real dist = abs(scores_unlabeled[clip] - eval_points[point]);
      if (dist < min_dist) {
        min_dist = dist;
        nearest_point_idx = point;
      }
    }
    num_clips[nearest_point_idx] = num_clips[nearest_point_idx] + 1;
  }
  for (point in 1 : N_points) {
    print("At point ", point, " there are ", num_clips[point], " clips");
    }
  print("Total:", sum(num_clips));
}
parameters {
  real<lower=0> rho;
  // real<lower=0> alpha;
  vector[xlen] eta;
}
transformed parameters {
  vector[xlen] f;
  {
    matrix[xlen, xlen] L_K;
    matrix[xlen, xlen] K = gp_exp_quad_cov(x, 1, rho); 
    
    // diagonal elements
    for (n in 1 : xlen) {
      K[n, n] = K[n, n] + 1e-9; // add small value to diag to ensure positive definiteness
    }
    
    L_K = cholesky_decompose(K);
    f = L_K * eta; 
    for (n in 1 : xlen) {
      f[n] = x[n] + f[n]; // shift the mean to the score (i.e. prior is logits are calibrated)
    }
  }
}
model {
  // use a inv_gamma with mean 
  rho ~ normal(1, sd_scores/2);
  // alpha ~ normal(0, 3);
  eta ~ normal(0, 1);
  labels ~ bernoulli_logit(f[1 : N_labeled]);
}
generated quantities {
  int n_pos_pred = sum(labels);
  // times the probability at each inducing_point by its inducing_value
  {
    array[N_points] int pos_at_points;
    for (point in 1 : N_points) {
      if (num_clips[point] == 0) {
        pos_at_points[point] = 0;
      } else {
        pos_at_points[point] = binomial_rng(num_clips[point], inv_logit(f[point]));
      }
    }
    n_pos_pred = n_pos_pred + sum(pos_at_points);
  }
}