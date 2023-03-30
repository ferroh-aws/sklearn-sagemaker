FROM public.ecr.aws/docker/library/python:3.11.2-slim-bullseye
ENV SAGEMAKER_SKLEARN_VERSION=1.2.2

LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true

RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get -y install --no-install-recommends \
        build-essential

COPY requirements.txt /requirements.txt
RUN python -m pip install -r /requirements.txt && \
    rm /requirements.txt

COPY dist/sklearn_container-1.0-py3-none-any.whl /sklearn_container-1.0-py3-none-any.whl
RUN pip install --no-cache /sklearn_container-1.0-py3-none-any.whl && \
    rm /sklearn_container-1.0-py3-none-any.whl

ENV SAGEMAKER_TRAINING_MODULE sklearn-container.training:main
ENV SAGEMAKER_SERVING_MODULE sklearn-container.serving:main

ENV SM_INPUT /opt/ml/input
ENV SM_INPUT_TRAINING_CONFIG_FILE $SM_INPUT/config/hyperparameters.json
ENV SM_INPUT_DATA_CONFIG_FILE $SM_INPUT/config/inputdataconfig.json
ENV SM_CHECKPOINT_CONFIG_FILE $SM_INPUT/config/checkpointconfig.json

# Set SageMaker serving environment variables
ENV SM_MODEL_DIR /opt/ml/model

EXPOSE 8080
ENV TEMP=/home/model-server/tmp

# Required label for multi-model loading
LABEL com.amazonaws.sagemaker.capabilities.multi-models=false