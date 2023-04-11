require(lme4)
library(lmerTest)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Test whether SPP has higher rate of non-mismatch-resolving responses
exp_names <- list("exp1", "exp2")
for (exp_name in exp_names){
  print(exp_name)
  data <- read.csv(file = paste('../data/', exp_name, '_df_target_trials.csv', sep=''))
  data$is_plural_attractor <- ifelse(startsWith(data$trial_condition, 'SPP'), 1, -1)
  data$non_mismatch_resolving_correction <- ifelse(data$e_type == 'subj' | data$e_type == 'verb' | data$e_type == 'subj+local' | data$e_type == 'verb+local', 0, 1)
  exp_data <- data
  model_glmer <- glmer(formula = "non_mismatch_resolving_correction ~ is_plural_attractor + (is_plural_attractor | item) + (is_plural_attractor | subject)",
                      data=exp_data, family=binomial(link="logit"))
  rs <- anova(model_glmer)
  print(rs)
  print(summary(model_glmer))
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

model_glmer <- glmer(formula = "non_mismatch_resolving_correction ~ is_optional_correction + (is_optional_correction || item) + (is_optional_correction || subject)",
                   data=data, family=binomial(link="logit"))
print(summary(model_glmer))