FROM petronetto/docker-python-deep-learning

WORKDIR /code
ADD bot.py api.py entrypoint.sh requirements.txt labels.txt /code/
ADD ./bin/* /code/bin/
ADD ./cfg/* /code/cfg/

RUN apt-get update -y && apt-get install -y --no-install-recommends ${BUILD_PACKAGES}

RUN apt-get install -y ssh

RUN git clone https://github.com/thtrieu/darkflow.git /root/darkflow && \
    cd /root/darkflow && \
    pip install --upgrade cython && \
    python setup.py build_ext --inplace && \
    pip install . && \
    cd -

RUN pip install -r requirements.txt
