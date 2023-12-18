FROM nvcr.io/nvidia/pytorch:22.12-py3

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update
RUN apt-get install -y zip htop screen libgl1-mesa-glx xcb

COPY requirements.txt /requirements.txt

RUN pip install -r /requirements.txt
