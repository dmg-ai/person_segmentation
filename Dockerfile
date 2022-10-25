FROM python
COPY . /person_segmentation

#RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
# RUN apt-get update
# RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -U pip
RUN pip install -U setuptools
# RUN apt-get install -y libxrender-dev

WORKDIR /person_segmentation
RUN pip install -r requirements.txt