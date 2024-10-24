# Committee  

__Author:__ Eliot Liucci  
__Chair:__ John Smith  
__Additional Members:__ Samidha Shetty, Katharine Banner, and Scott McCalla

# Directory Guide

- __Code__ 
  - __Comparison.py__: Used to compare LSTM to ARIMA and HW
  - __CPU\_vs\_GPU.py__: Test GPU vs. CPU performance for matrix multiplication
  - __Figures.R__: Script for creating plots in `ggplot2`
  - __GA\_LSTM.py__: Attempt at a genetic algorithm for determining architecture
  - __Hyperparameter\_Tuning.py__ Using Bayesian Optimization for tuning, training, and saving models on various data sets
  - __Interpolation.R__: Using Kalman filter to impute missing values in time series
  - __Pred\_Intervals.py__: Uses bootstrapped training data to obtain prediction intervals for LSTM
  - __Simulating\_TS.R__: Simulates ARIMA and SSM data and saves to `data` folder
- __Data__
  - __P33.csv__: EVER hydrostation Data used for initial model
  - __Pred\_Intervals.csv__: Contains predictions for bootstrapped LSTMs in `Pred_Intervals.py` (Most recent from Tempest Cluster)
  - __Shark\_Slough.csv__: Alternate EVER hydrostation data
  - __Simulated\_ARIMA\_TS.csv__: Simulated ARIMA data from `Simulating_TS.R`
  - __Simulated\_SSM\_TS.csv__: Simulated SSM data from `Simulating_TS.R`
- __Deliverables__
  - __Presentation__: Writing project presentation for MSU 2024
  - __Written Component__: Written paper exploring LSTMs
- __Model Diagnostics__: Graphics showing incremental improvement in training process from `Hyperparameter_Tuning.R`
- __Models__: Optimized and trained models from `Hyperparameter_Tuning.R`
- __Texts__: Supplemental texts

# To-Dos
  - John's comments in Overleaf
  - Discussion of SSM & ARIMA results and performance
    - ~~Introduce Simulation Methods & Cite Paper~~
      - ~~Kitagawa et al for SSM~~
      - ~~Introductory Time Series with R for ARIMA~~
    - Generate graphics
    - ~~Latin Hypercube for Hyperparameter Searching~~
  - ~~Equation tables~~
    - ~~Vertical buffer~~
    - ~~Third column `purpose'~~
  - ~~Increase number of features for parallel training~~
  - ~~Justify use of Kalman filter with reference paper~~
  - ~~Go to RCI Office Hours~~
  - ~~Finish Rough Draft~~
  - ~~Committee form in MyInfo~~
  - ~~Program of Study Form~~
  - ~~Find Meeting Time for Spring 2024~~
  - ~~Reach out to Kathi Irvine regarding use of GRA data~~ [Confirmed, Jan 4th, 2024]


# Tempest Commands

## Activate Mamba Env
mamba activate <env-name>
mamba deactivate <env-name>

## Loading Modules
module avail # shows installed modules
ms <program-name> # search for a module
module load <module-name> # enable a module

## Basic Linux Commands
pwd # print location in file system
ls # list files in current directory
mkdir dir_name # create new directory
cd dir_name # move to new directory
cd /path/to/directory # move to new location
nano filename # change file name

## Jobs
sqf
sqf -u
tail -f <output-file>
scontrol show jobid -dd <jobid>

# Potential Paper Framework

- Introduction
  - Background on NNs and RNNs
  - Background on LSTM
  - Motivation
    - Introduce EVER Data
- Methodology
  - Why LSTM?
  - Benefits of RNNs over traditional ARMA models
- Results
  - Forecasting accuracy
  - Reflection/Additional Applications

# Meeting Notes

# 9/17/2024

## Notes
- Reach out to tempest folks about setting up venv for bootstrap intervals
  - Request to go under Johns partition
- Test model on simulated time-series
  - Simulate using **SARIMA** or State-Space

# 2/29/2024  

## Notes  
- Reference specific examples in first couple of slides
- Motivate ARIMA as historically common method for forecasting
- New Outline
  - Hook with time-series facts
  - Reference specific instances of time-series data
    - Yellowstone flood
    - Gamestop stock
  - RNNs and their use for time-series forcasting
    - LSTM Networks
  - Data Available, motivating question
    - Missingness, how we dealt with it
  - Results
    - LSTM vs. ARIMA
  - Pros and Cons
  - Next Steps
    - Compare to Holt-Winters and Exponential Smoothing
- Kalman Filter for Interpolation
- Poster
  - Everything in presentation
  - Assumptions made for each method
  - Equations

# 2/1/2024  

## Notes  
- Outline looks reasonable
  - Introduction: Introduce Data and Motivation
    - Briefly, why are RNNs better than ARMA models?
  - Methodology: Introduce RNNs and LSTM
    - Math behind RNNs and LSTM
  - Discussion: Compare ARMA predictions to RNN predictions

- March Presentation
  - Motivation
  - Problem
  - Results
    - MSE on 2020-2023 measurements b\w ARMA and RNN
  - Discussion
    - Interval Predicitions for RNN?
   
- Editing may spill over into Fall 2024 if need be

## To-Dos  
- Finish introduction
- Have some semblence of results for Wednesday, Feb 7th

# 1/9/2024

## Notes  
- Forecasting with RNN as First Project

## To-Dos  
- Discuss with Katie Banner about filling in for Dominique as a Committee Member
- Create a Teams channel for communication
- Find a meeting time

