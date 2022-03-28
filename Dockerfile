FROM osgeo/gdal:ubuntu-small-3.4.1
ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

# Copying install_dependencies.sh & give permissions
COPY install_dependencies.sh /
RUN chmod +x /install_dependencies.sh

# install dependencies - making some changes here to test 
RUN ./install_dependencies.sh

RUN apt-get update

RUN /usr/bin/python3 -m pip install --upgrade pip
RUN apt-get install python3-cartopy -y
# install python packages - trial v0.5
COPY requirements.txt /
RUN pip3 --no-cache-dir install --upgrade setuptools && \
    pip3 --no-cache-dir install wheel && \
    pip3 --no-cache-dir install -r requirements.txt

RUN apt-get update  
RUN apt install saga -y

# Create directory to work
RUN mkdir dsm2dtm_dir

CMD ["/bin/bash"]
