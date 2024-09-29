## XXXXX VS-Code mapping data folders and configuring DVC

FROM tensorflow/tensorflow:latest-gpu-jupyter

ARG MY-RPOJECT-NAME
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
ADD ./requirements.txt .
RUN apt-get update && apt-get install -y git
RUN apt-get install -y openssh-client
RUN apt-get install sudo
RUN apt-get install wget
RUN pip3 -q install pip --upgrade
RUN pip install --upgrade pip

RUN pip install -r requirements.txt
RUN pip install dvc

#RUN nbdev_install_quarto
## Matching the cuda version for this container and T5 server Feb 2023
## COMMENTED OUT FOR THIS REPO, GOT could not find ERROR ##
# RUN pip install --upgrade torch==1.12.1 --extra-index-url https://download.pytorch.org/whl/cu113 

#TODO: How to change the project name dynamicaly
RUN git config --global --add safe.directory /workspaces/${MY-PROJECT-NAME}