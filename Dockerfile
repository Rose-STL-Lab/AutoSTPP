FROM gitlab-registry.nrp-nautilus.io/prp/jupyter-stack/minimal

USER root

# Install dependency
RUN apt update && apt install -y make rsync git s3cmd vim ffmpeg

# Add ssh key
RUN mkdir -p /root/.ssh
ADD .ssh/id_rsa /root/.ssh/id_rsa
ADD .ssh/config /root/.ssh/config
ADD .ssh/known_hosts /root/.ssh/known_hosts
RUN chmod 400 /root/.ssh/id_rsa

# Pull the latest project
WORKDIR /root/
RUN git clone --depth=1 ssh://git@gitlab-ssh.nrp-nautilus.io:30622/ZihaoZhou/autoint.git
WORKDIR /root/autoint/

RUN conda update --all
RUN conda install -c conda-forge conda-lock
RUN conda-lock install --name autoint-stpp
RUN conda clean -qafy
RUN mamba update -y -c conda-forge ffmpeg

# Activate the new conda environment
SHELL ["/opt/conda/bin/conda", "run", "-n", "autoint-stpp", "/bin/bash", "-c"]
RUN poetry install