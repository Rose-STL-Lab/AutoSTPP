FROM continuumio/miniconda3:4.12.0

USER root

# Install dependency
RUN mkdir -p /root/AI-STPP
WORKDIR /root/AI-STPP

ADD ./conda-linux-64.lock /root/AI-STPP/conda-linux-64.lock
ADD ./pyproject.toml /root/AI-STPP/pyproject.toml
ADD ./poetry.lock /root/AI-STPP/poetry.lock
RUN conda create --name autoint --file conda-linux-64.lock
RUN conda clean -afy

# Activate the new conda environment
SHELL ["conda", "run", "-n", "autoint", "/bin/bash", "-c"]
RUN poetry install

# Install make
RUN apt update && apt install -y make

# Setup Github SSH private key
ADD ./id_rsa /root/.ssh/id_rsa
RUN chmod 600 /root/.ssh/id_rsa
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts

# Pull the latest project
RUN git clone git@github.com:Rose-STL-Lab/AI-STPP.git temp --branch 0.1.0
RUN mv temp/* .
RUN rm -rf temp/
