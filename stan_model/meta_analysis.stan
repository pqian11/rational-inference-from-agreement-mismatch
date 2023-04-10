data {
  int<lower=0> N;         // number of studies
  int<lower=0> K;         // number of conditions
  int<lower=0> J;         // number of inferred error types
  int error_freq[N, K, J];	      // observed error count
}
parameters {
  simplex[J] mu[K];
  real<lower=0> sigma;
  vector[J] b_exp_unit[N, K];      // non-centered parameterization
}
transformed parameters {
  simplex[J] theta[N, K];
  vector[J] b_exp[N, K];
  vector[J] log_theta[N, K];

  for (n in 1:N){
    for (k in 1:K){
      for (j in 1:J){
        b_exp[n, k, j] = sigma*b_exp_unit[n, k, j];
      }
    }
  }

  for (n in 1:N){
    for (k in 1:K){
     log_theta[n, k] = log(mu[k]) + b_exp[n, k]; 
     theta[n, k] = softmax(log_theta[n, k]);
    }
  }
}
model {
  sigma ~ normal(0, 100);

  for (n in 1:N){
    for (k in 1:K){
      for (j in 1:J){
        b_exp_unit[n, k, j] ~ normal(0, 1);
      }
    }
  }

  for (n in 1:N){
    for (k in 1:K) {
      error_freq[n, k] ~ multinomial(theta[n, k]);
    }
  }  
}
