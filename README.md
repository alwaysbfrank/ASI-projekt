# ASI_2022
 Architektury i metodologie wdrożeń systemów SI. 

Przed uruchomieniem kodów stwórz odpowiednie środowisko `conda`.

1. Install bundle `conda`
2. Download file `environment.yml`
3. Create environment: `$ conda env create -f environment.yml`
4. Activate Environment: `$ conda activate ASI-projekt`

# docker setup
1. docker build -f Dockerfile .
2. docker run {container id}

# application flow
1. Preparing initial data (separate batches to simulate the incoming flow of new data)
2. Training the data model using logistic regression.
3. Checking the drift on two predicted data sets (on two metrics each - r2 and rbms for new batch)
4. If there is a drift, train a new model on all the data that has come in so far.

# structure
run structure -> docker builds an image using miniconda3 which installs python and other dependencies


