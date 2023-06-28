#py37
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel
RUN pip install --upgrade pip
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    sudo wget screen unzip less nano
WORKDIR /workspace
COPY requirement.txt .
RUN pip install -r requirement.txt

# for opencv
RUN apt-get install -y ffmpeg libsm6 libxext6