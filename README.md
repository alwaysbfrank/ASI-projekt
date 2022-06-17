# ASI-projekt

## run with docker
1. build container `docker build -f Dockerfile .`
2. run container `docker run {container id}`

## run with conda
1. create environment `conda env create -f environment.yml`
2. activate environment `conda activate ASI-projekt`
3. run app `python main.py`

## application flow
1. Preparing initial data (separate batches to simulate the incoming flow of new data).
2. Training the data model using logistic regression.  
3. For each new batch:
   1. Check if there is a drift on two predicted data sets (on two metrics each - r2 and rbms).
   2. Add the new data to training dataset.
   3. If there is a drift, train a new model on all the data that has come in so far.

## structure
run structure -> docker builds an image using miniconda3 which installs python and other dependencies.