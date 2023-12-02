## Start from this Docker image
FROM nvidia/cuda:11.3.0-base-ubuntu20.04


## Set workdir in Docker Container
# set default workdir in your docker container
# In other words your scripts will run from this directory
# RUN mkdir /workdir
WORKDIR /home/wmaag/UROP3100/Docker_submission

## Copy your files into Docker Container
COPY ./ /home/wmaag/UROP3100/Docker_submission
RUN chmod a+x /home/wmaag/UROP3100/Docker_submission/Docker_infer.py

## Install requirements
RUN conda env create -f semi_cmr.yaml
RUN pip
## Make Docker container executable
ENTRYPOINT ["/opt/conda/bin/python", "Docker_infer.py"]
