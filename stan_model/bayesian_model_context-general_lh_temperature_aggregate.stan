data {
  int<lower=0> N;           // number of items
  int<lower=0> K;           // number of conditions
  int<lower=0> J;           // number of inferred error types
  real obs_e_prob[N, K, J]; // observed edit type distribution
  real prior[N, K];	        // prior over the conditions
}
parameters {
  simplex[J] error_prob_vec;
  real<lower=0> a;
  real<lower=0> sigma;      // standard deviation of the noise term
}
transformed parameters {
  vector[J] inferred_e_prob[N, K];
  simplex[J] error_prob[K];
  
  error_prob[1] = error_prob_vec;
  error_prob[2] = error_prob_vec;
  error_prob[3] = error_prob_vec;
  error_prob[4] = error_prob_vec;

  for (n in 1:N){
    for (k in 1:K){
      if (k == 1){
        inferred_e_prob[n, k, 1] = log(error_prob[3, 1]) + log(prior[n, 3])*a;
        inferred_e_prob[n, k, 2] = log(error_prob[1, 2]) + log(prior[n, 1])*a;
        inferred_e_prob[n, k, 3] = log(error_prob[4, 3]) + log(prior[n, 4])*a;
        inferred_e_prob[n, k, 4] = log(error_prob[2, 4]) + log(prior[n, 2])*a;
      }else if (k == 2){
        inferred_e_prob[n, k, 1] = log(error_prob[4, 1]) + log(prior[n, 4])*a;
        inferred_e_prob[n, k, 2] = log(error_prob[2, 2]) + log(prior[n, 2])*a;
        inferred_e_prob[n, k, 3] = log(error_prob[3, 3]) + log(prior[n, 3])*a;
        inferred_e_prob[n, k, 4] = log(error_prob[1, 4]) + log(prior[n, 1])*a;
      }else if (k == 3){
        inferred_e_prob[n, k, 1] = log(error_prob[1, 1]) + log(prior[n, 1])*a;
        inferred_e_prob[n, k, 2] = log(error_prob[3, 2]) + log(prior[n, 3])*a;
        inferred_e_prob[n, k, 3] = log(error_prob[2, 3]) + log(prior[n, 2])*a;
        inferred_e_prob[n, k, 4] = log(error_prob[4, 4]) + log(prior[n, 4])*a;
      }else {
        inferred_e_prob[n, k, 1] = log(error_prob[2, 1]) + log(prior[n, 2])*a;
        inferred_e_prob[n, k, 2] = log(error_prob[4, 2]) + log(prior[n, 4])*a;
        inferred_e_prob[n, k, 3] = log(error_prob[1, 3]) + log(prior[n, 1])*a;
        inferred_e_prob[n, k, 4] = log(error_prob[3, 4]) + log(prior[n, 3])*a;
      }
      inferred_e_prob[n, k] = softmax(inferred_e_prob[n, k]);
    }
  }
}
model {
  for (n in 1:N){
    for (k in 1:K) {
      for (j in 1:J){
        obs_e_prob[n, k, j] ~ normal(inferred_e_prob[n, k, j], sigma);
      }
    }
  }  
}
generated quantities {
  real log_lh = 0;
  for (n in 1:N){
    for (k in 1:K) {
      for (j in 1:J){
        log_lh += normal_lpdf(obs_e_prob[n, k, j] | inferred_e_prob[n, k, j], sigma);
      }
    }
  }
}
