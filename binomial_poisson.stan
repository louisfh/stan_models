data {
    int<lower=0> N; // number of trials
    array[N] real<lower=0, upper=1> p; // probability of success
}
parameters {
}
model {
}
generated quantities {
    int<lower=0> n_pos;
    int<lower=0> n_pos_pred;

    n_pos = 0;
    for (i in 1:N) {
        n_pos = n_pos + bernoulli_rng(p[i]);
    }
    {
        array[N] real p_mod; //no constraints allowed on local!
        p_mod = p;
        // add 0.01 to each element of p_mod
        for (i in 1:N) {
            p_mod[i] = p_mod[i] + 0.01;
        }
        n_pos_pred = sum(bernoulli_rng(p_mod));
    }
}
