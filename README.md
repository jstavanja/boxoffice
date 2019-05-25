# boxoffice

This repository contains our code, used in the 
[TMDB Box Office Prediction](https://www.kaggle.com/c/tmdb-box-office-prediction/) competition.  
The data was obtained from TMDB.

## Setup
[Optional] Create a virtual environment.
```
$ virtualenv --python=python3 --system-site-packages boxofficevenv
$ source boxofficevenv/bin/activate
```

Then, install the required dependencies.
```
$ pip3 install -r requirements.txt
```

**Note**: you might need to replace `pip3` with `pip`.  
**Note 2**: this does not mean the code is *Python2.x*-compatible 
(if it is, it's completely coincidental).

## Data
`train.csv` and `test.csv` contain the training and test data.  
`fixes_train_budget_revenue.json` contains fixed budgets and revenues for some movies (keys 
represent `imdb_id` of movies) in the training set.  
`fixes_genre.json` contains genre information for the examples in training and test set which do
not have it.

## Results
Using root mean squared logarithmic error (RMSLE) as it is used for submission scoring on Kaggle.
The approaches are briefly described in the next section.  

| Approach  	  | Offline  | Kaggle submission |  
|:---------------:|:--------:|:-----------------:|  
| Baseline  	  |  3.70140 | 3.73362           |  
| Basic features  |  2.34904 | 2.88622           |  
| All features    |  1.55815 | 2.52469           |
| 4 Features      |  1.55607 | 2.51451           |

## Approaches
`Baseline`:  predict mean revenue of movies in training set for the movies of test set.
70%-30% train-test split used for offline evaluation.

`Basic features`: use 3 basic features: genres, original_language and runtime. 
Encode genres and original_language using 1-hot-encoding. Add new feature (count 
of genres provided for film). Group less frequent original_languages (< 5 occurences
in training set) under 'other' attribute.
70%-10%-20% train-val-test-split used for offline evaluation. Best score obtained 
with XGBoost.

`All features`: use all of the features: runtime, popularity, budget (excluded budgets < 100),
original_language, spoken_langugate, important_cast_count, top_director/writer/producer, 
production_companies/countries_count, is_weekend, release_weekday

`All features`: use four original features: popularity, runtime, budget (excluded budgets < 100),
and genres.