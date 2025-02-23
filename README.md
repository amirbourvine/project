OVERVIEW OF GIT STRUCTURE:
  - gymnasium_env: python package (we created) for gym envs, which includes the ElectricityMarket environment.
    
  - DDPG_forecasting: a directory that includes the third (research) part of the project.
    
      * DDPG_agent.py: includes main logic for DDPG agent
      * DataPreprocessing.py: includes pre-processing stages of the data (from .csv to train)
      * all csv files include datasets. V6.csv is custom (not related to the project), all other include env_type (explained in config.py), either p or d and number of features for               forecast mode.
      * main.py is custom (not related to this project)
      * forecast_agent.py: includes the agent for the ElectricityMarket environment that uses the forecasting models
      * forecast_eval_log.txt: includes logs of a single training of the forecast model (from forecast_agent.py)
      * gen_dataset.py: logic to generate the datasets for our environments
      * main_forecast.py: file to run to train and evaluate the forecast agent
      * train_forecast.py: logic to train the forecast models
    
  - requirements.txt: dependencies needed to be installed for the project.
    
  - config.py: variables for all of the project.
    
  - env_demo: basic checks for the environment.
    
  - eval_model: evaluation of a stable_baseline3 model
    
  - main.py: script to run for the second part of the project to train and evaluate stable_baseline3 model
    
  - tensorboard_logs: saved metrics from the second part of the project
