FROM tensorflow/tensorflow:1.9.0-py3


RUN apt-get update -yqq \
    && apt-get install -y locales\
    && apt-get install -yqq \
    && pip3 install --upgrade pip \
    && locale-gen en_US.UTF-8

RUN pip3 install keras
RUN pip3 install gensim

COPY stock_trading Chapter09
COPY chatbot Chapter07
COPY Chapter02 Chapter02