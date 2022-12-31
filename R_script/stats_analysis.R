library(brms)
require(lme4)
library(lmerTest)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

exp_names <- list("exp1", "exp2")
for (exp_name in exp_names){
  print(exp_name)
  data <- read.csv(file = paste('../data/', exp_name, '_df_target_mismatch_resolving_trials.csv', sep=''))
  data$is_N1_S <- ifelse(startsWith(data$trial_condition, 'S'), 1, -1)
  data$is_attractor <- ifelse(startsWith(data$trial_condition, 'SP') | startsWith(data$trial_condition, 'PS'), 1, -1)
  data$is_plural_attractor <- ifelse(startsWith(data$trial_condition, 'SPP'), 1, -1)
  data$subj_error <- ifelse(data$e_type == 'subj', 1, 0)
  # data$subj_error <- ifelse(startsWith(data$e_type, 'subj'), 1, 0)
  model_lmer <- lmer(formula = "subj_error ~ is_N1_S + is_attractor + is_plural_attractor + (is_N1_S + is_attractor + is_plural_attractor | item) + (is_N1_S + is_attractor + is_plural_attractor | subject)",
                      data=data)
  rs <- anova(model_lmer)
  print(rs)
}

# model1 <- brms::brm(formula = "subj_error ~ is_N1_S + is_attractor + is_plural_attractor + (is_N1_S + is_attractor + is_plural_attractor || item) + (is_N1_S + is_attractor + is_plural_attractor || subject)",
#                     data=exp1_data, family='bernoulli', iter=10000, cores=4, chains=4)
# 
# 
# model2 <- brms::brm(formula = "subj_error ~ is_N1_S + is_attractor + is_plural_attractor + (is_N1_S + is_attractor + is_plural_attractor || item) + (is_N1_S + is_attractor + is_plural_attractor || subject)",
#                     data=exp2_data, family='bernoulli', iter=10000, cores=4, chains=4)


# Test whether SPP has higher rate of non-mismatch-resolving responses
exp_names <- list("exp1", "exp2")
for (exp_name in exp_names){
  print(exp_name)
  data <- read.csv(file = paste('../data/', exp_name, '_df_target_trials.csv', sep=''))
  data$is_plural_attractor <- ifelse(startsWith(data$trial_condition, 'SPP'), 1, -1)
  data$non_mismatch_resolving_correction <- ifelse(data$e_type == 'subj' | data$e_type == 'verb' | data$e_type == 'subj+local' | data$e_type == 'verb+local', 0, 1)
  exp_data <- data
  model_lmer <- lmer(formula = "non_mismatch_resolving_correction ~ is_plural_attractor + (is_plural_attractor | item) + (is_plural_attractor | subject)",
                      data=exp_data)
  rs <- anova(model_lmer)
  print(rs)
}

# Test whether making the correction optional in exp2 increases
# the rate of non-mismatch-resolving responses
data1 <- read.csv(file = '../data/exp1_df_target_trials.csv')
data2 <- read.csv(file = '../data/exp2_df_target_trials.csv')
data1$subject <- paste('exp1_', data1$subject, sep='')
data2$subject <- paste('exp2_', data2$subject, sep='')
data1$is_optional_correction = -1
data2$is_optional_correction = 1
data <- rbind(data1, data2)

data$non_mismatch_resolving_correction <- ifelse(data$e_type == 'subj' | data$e_type == 'verb' | data$e_type == 'subj+local' | data$e_type == 'verb+local', 0, 1)

# model_lmer <- lmer(formula = "non_mismatch_resolving_correction ~ is_optional_correction + (is_optional_correction | item) + (is_optional_correction | subject)",
#                    data=data)

model_lmer <- lmer(formula = "non_mismatch_resolving_correction ~ is_optional_correction*trial_condition + (1 + trial_condition + is_optional_correction | item) + (1 + trial_condition + is_optional_correction | subject)",
                   data=data)
rs <- anova(model_lmer)
print(rs)
