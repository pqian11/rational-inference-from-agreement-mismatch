# Comprehenders’ Error Correction Mechanisms are Finely Calibrated to Language Production Statistics

This repository accompanies a study of how humans correct subject-verb number agreement mismatch in English.

## Environment
Use `stan_env.yml` to install the `conda` environment. Run the following code to download the `nltk` word tokenizer.
```
import nltk
nltk.download('punkt')
```
Download the [data folder](https://osf.io/6hq8e/) and put it in the root of the repository. Create the following directory structure from the root folder:
```
mkdir -p fig
mkdir -p stan_model/model_fit
```

## Data analysis
Estimate prior from norming task data.
```
python src/estimate_prior_from_norming_data.py
```

Export relevant data frame of Study I and II for visualization, statistical analysis, and modelling work.
```
python src/export_df
```

Analyze human language production experiments from [Ryskin et al. (2021)](https://psyarxiv.com/uaxsq/) and [Kandel et al. (2022)](https://escholarship.org/uc/item/5wq6w93j) and output error counts for `SS_`, `SP_`, `PS_`, and `PP_`. The data is used to estimate the likelihood of different type of errors given an intended message in language production and further compared to the error probabilities inferred from the correction tasks.
```
python src/analyze_language_production_exps.py
```

Run a meta-analysis on data from the language production studies.
```
python src/meta_analysis.py
```

Bayesian estimation of the model parameters based on human data. Models are fitted on data from two studies respectively. `${MODEL_TYPE}` can take the value of `prior-only`, `lh-only`, `context-general_lh` and `full` , which corresponds to different versions of the ablated models as well as the full model described in the paper.
```
python src/fit_model_with_random_effects.py -m ${MODEL_TYPE}
```

Model fitting on combined data from both Study I&II.
```
python src/fit_model_with_random_effects_on_combined_data.py -m ${MODEL_TYPE}
```

Display summary statistics of the Bayesian estimation results for selected model parameters and visualize estimated error likelihood parameters for  `lh-only`, `context-general_lh` and `full`  models.
```
python src/model_rs_analysis.py
```

Run maximum likelihood estimation (MLE) and display likelihood ratio test results between `context-general_lh` and `full`  models.  `--do_compile` flag can be omitted to load previously compiled model.
```
python src/get_lrt_rs.py
```

## Visualization
Visualize the distribution of edit types given trial conditions for two free-form error-correction tasks respectively.
```
python src/plot_e_dist_in_human_data.py
```

Visualize the effect of prior by comparing the estimated prior of the observed subject phrase to the frequency of verb correction for the specific trial condition of an item.
```
python src/plot_verb_correction_freq_against_prior.py
```

Visualize overall comparison between human correction preferences and model correction preferences.
```
python src/plot_model_human_comparison.py
```

Visualize and compare error likelihood estimated from language comprehension tasks (free-form editing in Study I and II) and language production tasks.
```
python src/plot_error_lh.py
```

## Statistical tests
`R_script` folder contains `R` script for mixed-effect regression analysis.


## Supplemental analyses
Use pretrained language model (GPT-2) to estimate the prior probabilities of the intend messages (require installation of `transformers` package.
```
python script/get_gpt2_prior_norm.py
```

Estimate likelihood parameters in the Bayesian model using human data from the error correction studies together with prior probabilities derived from GPT-2. Models are fitted on data from two studies respectively. `${MODEL_TYPE}` can take the value of `prior-only`, `lh-only`, `context-general_lh` and `full` , which corresponds to different versions of the ablated models as well as the full model described in the paper.
```
python src/fit_model_with_random_effects_and_gpt2_norm.py -m ${MODEL}
```

Fit models with human behavioral data combined from Study I&II as well as prior probabilities derived from GPT-2.
```
python script/fit_model_with_random_effects_and_gpt2_norm_on_combined_data.py -m ${MODEL}
```

Plot a visual comparison between human-normed priors and priors derived from GPT-2.
```
python script/compare_human_lm_priors.py
```

Run a random split-half analysis on combined error correction data from Study I and II. The combined data is randomly split into two halves multiple times. For each random split, we compute the Mean Squared Error and Spearman correlation coefficient between error correction frequency estimated from each half. Then we plot the distribution of MSE and Spearman correlation coefficients across many random splits.
```
python script/random_split_half_analysis.py
```
