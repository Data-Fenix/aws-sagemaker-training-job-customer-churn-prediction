# Base debian system
FROM ubuntu:20.04
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Update OS
RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1


RUN apt-get update && apt-get install -y --no-install-recommends \
     python3 \
     python3-pip \
     python3-dev \
     && \
     apt-get clean && \
     rm -rf /var/lib/apt/lists/*

RUN apt-get update --fix-missing && \
    apt-get -y install gcc g++ make 
    
ARG PIP=pip3

WORKDIR /
    
# Installing lightgbm, tqdm
RUN ${PIP} install --no-cache --upgrade \
            lightgbm \
            tqdm \
            xgboost \
            seaborn \
            sklearn \
            pyarrow \
            fastparquet
            
# Setting some environment variables.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib" \
    PYTHONIOENCODING=UTF-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN ${PIP} install --no-cache --upgrade \
    sagemaker-training

# Copies code under /opt/ml/code where sagemaker-containers expects to find the script to run
COPY model/train.py /opt/ml/code/
#COPY regressor_model/aws_helper.py /opt/ml/code/

# Defines train.py as script entry point
ENV SAGEMAKER_PROGRAM train.py
# ENTRYPOINT ["python", "/opt/ml/code/train.py"]