FROM debian:8.4

# base dependencies, hardcoding the mirror due to apt-get problems
# cf. issue https://stackoverflow.com/questions/32304631/docker-debian-apt-error-reading-from-server
RUN export DEBIAN_FRONTEND=noninteractive; \
    sed --in-place 's/httpredir.debian.org/mirror.sov.uk.goscomb.net/' /etc/apt/sources.list && \
    apt-get update --fix-missing && \
    apt-get -y upgrade && \
    apt-get install -y wget bzip2 ca-certificates \
    curl grep sed git build-essential\
    nginx gettext-base \
    supervisor # this uses system python 2.7

# Ngnix setup adapted from https://github.com/tiangolo/uwsgi-nginx-docker/blob/master/python3.5/Dockerfile 
# forward request and error logs to docker log collector
RUN ln -sf /dev/stdout /var/log/nginx/access.log &&\
    ln -sf /dev/stderr /var/log/nginx/error.log &&\
    echo "daemon off;" >> /etc/nginx/nginx.conf &&\
    rm /etc/nginx/sites-enabled/default


COPY "build_tools/requirements_*.txt" /tmp/


# Install Anaconda Python 3 distribution (for Python, R)
# and all the dependencies
# this is more reliable than using system python
RUN wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh  && \
    chmod +x miniconda.sh && \
    ./miniconda.sh -b && \
    export PATH=/root/miniconda3/bin:$PATH && \
    conda update --yes --no-deps conda && \
    conda config --set always_yes yes --set changeps1 no && \
    conda install --yes --file /tmp/requirements_conda.txt python=3.5 &&\
    pip install -r /tmp/requirements_pip_unix.txt && \
    pip install uwsgi==2.0.14 # to communicate with the nginx web-server 

# Copy the modified Nginx conf
COPY build_tools/docker/nginx.conf /etc/nginx/sites-enabled/

COPY build_tools/docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Copy the base uWSGI ini file to enable default dynamic uwsgi process number
COPY build_tools/docker/uwsgi.ini /etc/uwsgi/

ADD . /freediscovery_backend/

WORKDIR /freediscovery_backend/

ENV PATH /root/miniconda3/bin:$PATH

RUN python setup.py install && \
    cd /tmp/ && python -c "import freediscovery.tests as ft; ft.run()" 


# http://bugs.python.org/issue19846
# > At the moment, setting "LANG=C" on a Linux system *fundamentally breaks Python 3*, and that's not OK.
# rth: not sure if this is still relevant.
ENV LANG C.UTF-8


CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]

