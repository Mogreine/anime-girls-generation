FROM nvcr.io/nvidia/pytorch:21.07-py3

WORKDIR /app

COPY requirements.txt requirements.txt

# Activate conda environment for bash
RUN conda init bash

# Install other requirements
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -r requirements.txt
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 --upgrade

ENTRYPOINT [ "bash" ]
