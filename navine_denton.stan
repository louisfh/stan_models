data {
    int<lower=0> N_bins;
    array[N_bins] int<lower=0> num_clips; // number of clips in each bin
    array[N_bins] int<lower=0> num_labeled; // the number of clips labeled in each bin
    array[N_bins] int<lower=0> num_positives; // number of positive clips in each bin
}
transformed data {
    int<lower=0> N_positives = sum(num_positives);
    // calculate the number of unlabeled clips in each bin
    array[N_bins] int num_unlabeled;
    for (bin in 1:N_bins) {
        num_unlabeled[bin] = num_clips[bin] - num_labeled[bin];
    }
}
parameters {
    array[N_bins] real<lower=0, upper=1> p_pos; // probability of a clip being positive in each bin
}
model {
    p_pos ~ beta(1/3, 1/3); // prior
    for (bin in 1:N_bins) {
        target += binomial_lpmf(num_positives[bin] | num_labeled[bin], p_pos[bin]);
    }
}
generated quantities {
    int<lower=0> n_pos_pred;
    {
        array[N_bins] int pos_at_bins;
        for (bin in 1:N_bins) {
            pos_at_bins[bin] = binomial_rng(num_clips[bin], p_pos[bin]);
        }
        n_pos_pred = sum(pos_at_bins);
    }
}
