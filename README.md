
Name: Michele De Zotti
Student Number: 20260570

------------
WEEK 2 --

WHICH IS BETTER? 

My answer is that Logistic regression achieves 67.9% test accuracy with stable performance training and test results are nearly identical, showing no overfitting. The second model reaches 82.9% training accuracy but only 63.1% test accuracy, revealing  overfitting with a 20% gap. I'll personally choose logistic regression it has better test accuracy and generalizes reliably to new data.

WEEK 3 --

The cleaning process removed duplicate records and unnecessary columns, made category labels consistent, and treated invalid entries as missing. Test accuracy was lower for both models: the decision tree fell from 63.1% to 60.4%, while logistic regression went from 67.9% to 65.5%. Even after cleaning, logistic regression performed more consistently, with 67.8% training accuracy and 65.5% test accuracy, compared with 79.9% and 60.4% for the decision tree. Since cleaning changed which records ended up in the test set, these results are not a direct comparison, and the lower scores alone do not prove that cleaning harmed either model.

WEEK 4 --

| Model | Holdout accuracy | 5-fold CV accuracy |

| Dummy | 0.550 | 0.549 ± 0.000 |
| Logistic regression | 0.674 | 0.672 ± 0.013 |
| Decision tree | 0.591 | 0.610 ± 0.018 |
| Random forest | 0.644 | 0.650 ± 0.018 |

Logistic regression performed best and gave similar results with holdout and cross-validation. The dummy model only predicts the most common class, so its accuracy is a baseline, not useful detection of reoffending. The decision tree and random forest had larger gaps between training and validation scores, which suggests overfitting. The locked test set was not used in this comparison.

WEEK 5 --

Tuning is controlled by the `tuning` section in `config.yaml`. When enabled, Optuna evaluates each candidate with cross-validation of the complete preprocessing-and-model pipeline on development data only. Nested cross-validation repeats the search inside each outer training fold and evaluates on that fold's untouched validation rows; those out-of-fold predictions are used for the selected model's classification and fairness reports. The final search is then run on all development rows and its winning pipeline is refit there. The best trial's inner-CV score is useful for choosing parameters, but it is optimistic; report the nested-CV validation mean and standard deviation instead. The locked test set is never used for tuning or model selection.

Set `tuning.enabled: false` to run without hyperparameter search. Add a search space under `tuning.search_spaces` for each additional model type before enabling tuning for it.

Example from the configured decision-tree run (30 trials, five inner and outer folds; scores rounded to three decimals):

| Model | Default CV mean ± std | Tuned nested CV mean ± std | Best tuning score | Optimism | Chosen hyperparameters |
|---|---:|---:|---:|---:|---|
| Decision tree | 0.610 ± 0.018 | 0.672 ± 0.017 | 0.679 | +0.002 | `max_depth=6`, `min_samples_leaf=101`, `criterion=gini` |
| Logistic regression | 0.672 ± 0.013 | 0.673 ± 0.016 | 0.678 | +0.002 | `C=0.0715` |
| Random forest | 0.650 ± 0.018 | 0.679 ± 0.015 | 0.682 | +0.003 | `n_estimators=73`, `max_depth=20`, `min_samples_leaf=36`, `max_features=0.5` |

Tuning improved the tree’s mean accuracy by 0.062 and the forest’s by 0.029, both larger than their default fold-to-fold standard deviations; logistic regression changed by only 0.001. The tuned forest had the highest mean nested-CV accuracy in this run, though its 0.006 lead over the tree is smaller than either model’s fold-to-fold standard deviation. The forest selected a deeper tree depth than the standalone decision tree, while averaging many trees reduces variance. Best-trial scores are slightly higher than nested-CV results, as expected; report the nested-CV estimates. These are single seeded runs and may vary across package versions.

CHALLENGES ANSWERS --

1. **Tune logistic regression.** I searched `C` on a log scale from 0.0001 to 100. The selected value was `C=0.0715`. Nested CV was 0.673 ± 0.016, compared with default CV at 0.672 ± 0.013: a 0.001 increase, much smaller than the fold-to-fold variation. Logistic regression had less to gain because its simpler linear decision boundary was already performing close to its tuned result.

2. **Tune the random forest.** I used 15 trials and searched 30–100 trees, depth, minimum leaf size and maximum features. The selected settings were 73 trees, depth 20, `min_samples_leaf=36` and `max_features=0.5`. Nested CV was 0.679 ± 0.015, up from default CV at 0.650 ± 0.018; the best trial scored 0.682, an optimism of 0.003. The forest chose deeper trees than the standalone tree (depth 6 in the accuracy run). This is plausible because averaging many trees reduces variance, allowing each tree to be deeper without relying on one tree's unstable predictions.

3. **Tune preprocessing.** I added `prep__numeric__impute__strategy` with `median` and `mean` as choices. This remains leak-free because the imputer is inside the pipeline: it is fitted only on each training fold, then applied to that fold's validation rows. The final search selected `median`. Fitting preprocessing once on all development rows before cross-validation would leak information from validation folds. Caching is valid only when each cached preprocessor is fitted on that fold's training rows; because the imputation strategy is itself being tuned, cached results must also be kept separate for each strategy.

4. **Change the scoring metric.** I compared `accuracy` with `balanced_accuracy` using the same data split, seed, trial budget and expanded search space. Under accuracy, nested CV was 0.668 ± 0.016 and the selected settings were depth 14, `min_samples_leaf=115`, `criterion=entropy`, imputation `median`. Under balanced accuracy, nested CV was 0.664 ± 0.012 and the settings changed to depth 6, `min_samples_leaf=93`, `criterion=gini`, imputation `median`. The out-of-fold recall for class 1 (reoffended) changed from 0.586 to about 0.58, while class 0 recall rose from 0.735 to about 0.75; in this run, balanced accuracy did not improve positive-class recall.

  False-positive rates by race (accuracy → balanced accuracy): African-American 0.33 → 0.32; Caucasian 0.21 → 0.19; Hispanic 0.21 → 0.19; Other 0.21 → 0.18; Asian 0.14 → 0.14; Native American 0.17 → 0.17. Rates were generally lower or unchanged, but the African-American/Caucasian gap widened slightly. The Asian and Native American groups have small sample sizes, so their rates are particularly uncertain; these figures are a simple comparison, not a complete fairness audit.


------------
# Baseline Predictive Pipeline -- ETAI

This is the **starting point** for your semester project: a small but *complete* predictive pipeline -- every piece a real project needs (entry point, config, data loading, preprocessing, model, evaluation), just kept as simple as possible for now.

The task: predict two-year recidivism using ProPublica's COMPAS
dataset -- the data behind a real 2016 investigation into a risk-
assessment algorithm actually used by US courts to help inform bail and sentencing decisions. See `data/README.md` for the full problem description and a complete data dictionary before you start.

It has some **deliberately weak spots**. Part of your work this
semester is finding them and making them better -- see the pipeline progress table below, which tracks what changes and why as the weeks
go on.

## Project structure

```
.
├── main.py                # entry point: run the whole pipeline
├── config.yaml             # all tunable settings live here
├── requirements.txt
├── src/
│   ├── data.py             # loading
│   ├── preprocessing.py    # cleaning + train/test split
│   ├── model.py             # model construction
│   ├── tuning.py            # Optuna search + nested cross-validation
│   ├── evaluate.py         # accuracy metrics + fairness check
│   └── results.py          # saves each run's report to disk
├── results/                # created automatically -- one file per run (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md            # problem description + full data dictionary
```

## Pipeline progress

This table is updated after each practical class, so you can always see what changed in the pipeline and why -- it's a running log, not a fixed syllabus.

| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` |
| 5 | Hyperparameter tuning | Optuna tunes the selected model on development data using CV of the full pipeline; nested CV estimates the tuning procedure honestly; the locked test set remains untouched |

## Environment setup

You only need to do this once per machine.

### macOS / Linux
```bash
python3 -m venv venv                 # creates an isolated Python environment in a folder called "venv"
source venv/bin/activate             # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```

### Windows -- PowerShell
```powershell
python -m venv venv                  # creates an isolated Python environment in a folder called "venv"
venv\Scripts\activate                # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```
If PowerShell blocks the activation script, run this once first:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Windows -- cmd.exe
Same three steps as above, just with cmd's own activation command:
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

Once the environment is active you'll see `(venv)` at the start of your prompt. To leave it later, run `deactivate` (same command on every OS).

### Every time after the first

Creating the environment and installing packages only needs to happen once, ever. Every other time you sit down to work -- a new terminal window, the next practical class, tomorrow -- you don't repeat any of the steps above. From the project's root folder, you just need to:

**macOS / Linux**
```bash
source venv/bin/activate
python main.py
```

**Windows**
```powershell
venv\Scripts\activate
python main.py
```

That's it -- activate, then run. If you don't see `(venv)` at the start of your prompt, the environment isn't active and `python main.py` may use the wrong Python (or fail to find a package) entirely.

## Running the pipeline

With the environment active (see above), from the project's root
folder, on any OS:
```bash
python main.py
```

This loads `config.yaml`, loads and preprocesses the data, trains the model, and prints:
- **train accuracy and test accuracy, side by side.** Comparing the two is how you catch overfitting: if the model looks much better on the data it was trained on than on data it's never seen, it has memorised rather than learned something that generalises. 
- a classification report on the test set
- a false-positive-rate-by-race comparison between our model and
  COMPAS's own score

All of this is also saved to a timestamped file in `results/` (e.g.`results/run_20260916_143012.txt`), so it doesn't just scroll past in your terminal -- open it later, or change something in `config.yaml` (like the model type) and compare the new file to the last one.
`results/` is created automatically the first time you run the
pipeline, and isn't tracked in git (see `.gitignore`) since it's
generated output, not source.

You're free to improve on this structure or restructure it entirely -- what matters is that your project stays runnable end-to-end with a single command, and that each piece (data, preprocessing, model, evaluation) stays easy to find and change independently.

## Dataset

See `data/README.md`.
