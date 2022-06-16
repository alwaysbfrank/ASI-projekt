FROM continuumio/miniconda3

WORKDIR /app

# Create the environment:
COPY . .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "ASI-projekt", "/bin/bash", "-c"]

# The code to run when container is started:
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "ASI-projekt", "python", "main.py"]