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
    real<lower=0> sd_scores = sd(scores_unlabeled);
    real mu_guess = mean(scores_unlabeled);
}
parameters {
    ordered[2] mu;
    vector<lower=0, upper= 2*sd_scores>[2] sigma;
    real<lower=0, upper=1> theta; 
} 
model {
    mu[1] ~ normal(mu_guess - sd_scores, sd_scores);
    mu[2] ~ normal(mu_guess + sd_scores, sd_scores);
    // a good guess is the sd of the scores
    sigma ~ normal(sd_scores, sd_scores);

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
    // this will be the sum of n_pos and the predicted number of positive clips
    // the predicted positives are bernoulli trials with probability that the datapoint came from the positive distribution
    // do this in curly braces
    {
        array[N_unlabeled] real prob_pos; // no constraints allowed on local variables
        for (n_unlab in 1:N_unlabeled) {
            // p(pos) = theta * p(pos|score) / p(score)
            // p(data) = log_mix(theta, p(pos|score), p(neg|score))
            prob_pos[n_unlab] = exp(log(theta) + normal_lpdf(scores_unlabeled[n_unlab] | mu[2], sigma[2]) - log_mix(theta,
                            normal_lpdf(scores_unlabeled[n_unlab] | mu[2], sigma[2]),
                            normal_lpdf(scores_unlabeled[n_unlab] | mu[1], sigma[1])));
        }
        n_pos_pred = n_pos + sum(bernoulli_rng(prob_pos));
    }
}
