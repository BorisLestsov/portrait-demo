FROM continuumio/miniconda3
#FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

SHELL ["/bin/bash", "-c"]

LABEL maintainer="Evgeny Nizhibitsky <nizhib@github.com>"

# Install ubuntu packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        python3-pip \
        python3-setuptools && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install python packages


COPY env.yml .
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
RUN . ~/.bashrc
RUN conda env create --name pytorch12 --file=env.yml
RUN echo "conda activate pytorch12" >> ~/.bashrc
COPY requirements.txt .
RUN pip3 install -r requirements.txt
RUN . ~/.bashrc


# Copy project files

COPY . .

# Run the service

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
CMD bash -c '. ~/.bashrc && conda activate pytorch12 && python3 -u app.py'
