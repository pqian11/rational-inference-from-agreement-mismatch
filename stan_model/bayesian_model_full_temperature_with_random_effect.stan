data {
  int<lower=0> N;         // number of items
  int<lower=0> K;         // number of conditions
  int<lower=0> J;         // number of inferred error types
  int<lower=0> M;         // number of participants
  int<lower=0> L;         // number of trials
  int e_type_list[L];     // edit choice index for each trial
  int subject_list[L];    // subject index for each trial
  int item_list[L];       // item index for each trial
  int condition_list[L];  // mismatch condition index for each trial
  real prior[N, K];	      // prior over the conditions
  real lambda;            // hyper-parameter for prior over random effects
}
parameters {
  simplex[J] error_prob[K];
  vector[J] b_item_condition_unit[N, K];      // non-centered parameterization
  vector[J] b_subject_condition_unit[M, K];   // non-centered parameterization
  real<lower=0> sigma_item;
  real<lower=0> sigma_subject;
  real<lower=0> a;
}
transformed parameters {
  vector[J] log_inferred_e_prob[N, K];
  vector[J] inferred_e_prob_by_trial[L];

  vector[J] b_item_condition[N, K];
  vector[J] b_subject_condition[M, K];

  for (n in 1:N){
    for (k in 1:K){
      for (j in 1:J){
        b_item_condition[n, k, j] = sigma_item*b_item_condition_unit[n, k, j];
      }
    }
  }

  for (m in 1:M){
    for (k in 1:K){
      for (j in 1:J){
        b_subject_condition[m, k, j] = sigma_subject*b_subject_condition_unit[m, k, j];
      }
    }
  }

  for (n in 1:N){
    for (k in 1:K){
      if (k == 1){
        log_inferred_e_prob[n, k, 1] = log(error_prob[3, 1]) + log(prior[n, 3])*a;
        log_inferred_e_prob[n, k, 2] = log(error_prob[1, 2]) + log(prior[n, 1])*a;
        log_inferred_e_prob[n, k, 3] = log(error_prob[4, 3]) + log(prior[n, 4])*a;
        log_inferred_e_prob[n, k, 4] = log(error_prob[2, 4]) + log(prior[n, 2])*a;
      }else if (k == 2){
        log_inferred_e_prob[n, k, 1] = log(error_prob[4, 1]) + log(prior[n, 4])*a;
        log_inferred_e_prob[n, k, 2] = log(error_prob[2, 2]) + log(prior[n, 2])*a;
        log_inferred_e_prob[n, k, 3] = log(error_prob[3, 3]) + log(prior[n, 3])*a;
        log_inferred_e_prob[n, k, 4] = log(error_prob[1, 4]) + log(prior[n, 1])*a;
      }else if (k == 3){
        log_inferred_e_prob[n, k, 1] = log(error_prob[1, 1]) + log(prior[n, 1])*a;
        log_inferred_e_prob[n, k, 2] = log(error_prob[3, 2]) + log(prior[n, 3])*a;
        log_inferred_e_prob[n, k, 3] = log(error_prob[2, 3]) + log(prior[n, 2])*a;
        log_inferred_e_prob[n, k, 4] = log(error_prob[4, 4]) + log(prior[n, 4])*a;
      }else {
        log_inferred_e_prob[n, k, 1] = log(error_prob[2, 1]) + log(prior[n, 2])*a;
        log_inferred_e_prob[n, k, 2] = log(error_prob[4, 2]) + log(prior[n, 4])*a;
        log_inferred_e_prob[n, k, 3] = log(error_prob[1, 3]) + log(prior[n, 1])*a;
        log_inferred_e_prob[n, k, 4] = log(error_prob[3, 4]) + log(prior[n, 3])*a;
      }
    }
  }

  for (trial_idx in 1:L){
    inferred_e_prob_by_trial[trial_idx] = softmax(log_inferred_e_prob[item_list[trial_idx], condition_list[trial_idx]] + b_item_condition[item_list[trial_idx], condition_list[trial_idx]] + b_subject_condition[subject_list[trial_idx], condition_list[trial_idx]]);
  }
}
model {
  sigma_item ~ exponential(lambda);
  sigma_subject ~ exponential(lambda);

  for (n in 1:N){
    for (k in 1:K){
      for (j in 1:J){
        b_item_condition_unit[n, k, j] ~ normal(0, 1);
      }
    }
  }

  for (m in 1:M){
    for (k in 1:K){
      for (j in 1:J){
        b_subject_condition_unit[m, k, j] ~ normal(0, 1);
      }
    }
  }

  for (trial_idx in 1:L){
    e_type_list[trial_idx] ~ categorical(inferred_e_prob_by_trial[trial_idx]);
  } 
}
generated quantities {
  real log_lh = 0;
  vector[J] inferred_e_prob[N, K];
  real delta_SP_SS = error_prob[2, 2] - error_prob[1, 2];
  real delta_PS_PP = error_prob[3, 2] - error_prob[4, 2];
  real delta_SP_PS = error_prob[2, 2] - error_prob[3, 2];

  for (n in 1:N){
    for (k in 1:K){
      inferred_e_prob[n, k] = softmax(log_inferred_e_prob[n, k]);
    }
  }

  log_lh += exponential_lpdf(sigma_item | lambda);
  log_lh += exponential_lpdf(sigma_subject | lambda);

  for (n in 1:N){
    for (k in 1:K){
      for (j in 1:J){
        log_lh += normal_lpdf(b_item_condition[n, k, j] | 0, sigma_item);
      }
    }
  }

  for (m in 1:M){
    for (k in 1:K){
      for (j in 1:J){
        log_lh += normal_lpdf(b_subject_condition[m, k, j] | 0, sigma_subject);
      }
    }
  }

  for (trial_idx in 1:L){
    log_lh += categorical_lpmf(e_type_list[trial_idx] | inferred_e_prob_by_trial[trial_idx]);
  }
}
