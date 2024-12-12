data {
    int<lower=0> N_unlabeled;
    int<lower=0> N_labeled;
    vector[N_unlabeled] scores_unlabeled;
    vector[N_labeled] scores_labeled;
    array[N_labeled] int<lower=0, upper=1> labels;
    vector[N_labeled] sampling_weight; // p(sampling)
}
transformed data {
    int<lower=0> total_clips = N_unlabeled + N_labeled;
    int<lower=0, upper=N_labeled> n_pos = sum(labels);
    real<lower=0> weights_mult; // used for re-weighting
    weights_mult = N_labeled / sum(sampling_weight);
}
parameters {
    ordered[2] mu;
    vector<lower=0>[2] sigma;
    real<lower=0, upper=1> theta; 
} 
model {
    for (i in 1:2) {
        mu[i] ~ normal(0, 3);
    }
    sigma ~ cauchy(0, 5);

    for (n_unlab in 1:N_unlabeled) {
            target += log_mix(theta,
                            normal_lpdf(scores_unlabeled[n_unlab] | mu[2], sigma[2]),
                            normal_lpdf(scores_unlabeled[n_unlab] | mu[1], sigma[1]));
                            }

    for (n_lab in 1:N_labeled) {
            if (labels[n_lab] == 0) {
                target += (log1m(theta) + normal_lpdf(scores_labeled[n_lab] | mu[1], sigma[1])) / (sampling_weight[n_lab]*weights_mult);
            } else { 
                target += (log(theta) + normal_lpdf(scores_labeled[n_lab] | mu[2], sigma[2])) / (sampling_weight[n_lab]*weights_mult);
            }
        }
}
generated quantities {
   int<lower=0, upper=total_clips> n_pos_pred;
    n_pos_pred = binomial_rng(total_clips, theta);
}
