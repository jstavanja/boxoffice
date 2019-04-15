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

## Results
Using root mean squared logarithmic error (RMSLE) as it is used for submission scoring on Kaggle.
The approaches are briefly described in the next section.  

| Approach  | Offline  | Kaggle submission |  
|:---------:|:--------:|:-----------------:|  
| Baseline  |  3.70140 | 3.73362           |  

## Approaches
`Baseline`:  predict mean revenue of movies in training set for the movies of test set.
70%-30% train-test split used for offline evaluation.
