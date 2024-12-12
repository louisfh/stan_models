data {
  int<lower=1> pop;                     // Population
  int<lower=1, upper=pop> n_sampled;    // Sample size
  int<lower=0, upper=n_sampled> n_pos;  // Number of positives observed
}
transformed data {
    // Max number of positives in population is pop - number of observed negatives
    int<lower=0> n_neg = n_sampled - n_pos;
    int<lower=0, upper=pop> max_a = pop-(n_neg);
    // Min number of positives is those observed in the sample
    int<lower=0, upper=n_sampled> min_a = n_pos;       
    int<lower=1, upper=pop> len_a = max_a-min_a+1; // numnber of possible values
    // array for marginalizing over  these
    array[len_a] int<lower=min_a, upper=max_a> marginalisation_array;
    for(a in 1:len_a) marginalisation_array[a] = min_a + a-1;
}
parameters {
}
transformed parameters {
    vector[len_a] lp;
    for(a in 1:len_a){
      lp[a] = hypergeometric_lpmf(n_pos | n_sampled, marginalisation_array[a], pop-marginalisation_array[a]) -log(len_a);
    } 
}
model {
    target += log_sum_exp(lp);
}
generated quantities{
  int i;
  simplex[len_a] tp;
  tp = softmax(lp);
  i = categorical_rng(tp);
}