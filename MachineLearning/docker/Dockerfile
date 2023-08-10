
FROM ubuntu:23.04

RUN apt update && \
    apt install -y vim python3-pip wget

RUN cd /tmp && \
    wget http://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz && \
    tar xvf install-tl-unx.tar.gz && \
    cd install-tl-2* && \
    perl ./install-tl --no-interaction
ENV PATH="$PATH:/usr/local/texlive/2023/bin/x86_64-linux"

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --break-system-packages -r /tmp/requirements.txt

ENTRYPOINT /bin/bash

