# Rational Inference from Agreement Mismatch

This repository accompanies a study of how humans correct subject-verb number agreement mismatch in English.

## Environment
Use `pystan_environment.yml` to install the `conda` environment. Run the following code to download the `nltk` word tokenizer.
```
import nltk
nltk.download('punkt')
```
Download the data folder and put it in the root of the repository. Create the following directory structure from the root folder:
```
mkdir -p fig
mkdir -p stan_model/pkl
mkdir -p stan_model/model_fit
```
To support maximal reproducibility of the computational modelling results, the definition file `script/stan_env.def` specifies the Singularity container that was used for fitting the Stan models.

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

Bayesian estimation of the model parameters based on human data. Models are fitted on data from two studies respectively. `${MODEL_TYPE}` can take the value of `prior-only`, `lh-only`, `context-general_lh` and `full` , which corresponds to different versions of the ablated models as well as the full model described in the paper. Adding `--do_compile`  recompiles the `stan` model file. If a model file is not modified, the `--do_compile` flag can be omitted to load previously saved compiled model.
```
python fit_model_with_random_effects.py -m ${MODEL_TYPE} --do_compile
```

Display summary statistics of the Bayesian estimation results for selected model parameters and visualize estimated error likelihood parameters for  `lh-only`, `context-general_lh` and `full`  models.
```
python src/model_rs_analysis.py
```

Run maximum likelihood estimation (MLE) and display likelihood ratio test results between `context-general_lh` and `full`  models.  `--do_compile` flag can be omitted to load previously compiled model.
```
python src/get_lrt_rs.py --do_compile
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

Visualize overall comparison between human correction preferences and model correction preferences:
```
python src/plot_model_human_comparison.py
```

Visualize and compare error likelihood estimated from language comprehension tasks (free-form editing in Study I and II) and language production tasks:
```
python src/plot_error_lh.py
```

## Statistical tests
`R_script` folder contains `R` script for linear mixed-effect model analysis.