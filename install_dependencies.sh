#!/bin/bash

echo "

This script is for installing dependencies for doktarpy

"

# Install main dependencies
package_list=(
    
    build-essential
    software-properties-common
    python3.8
    python3-tk
    python3-pip
    build-essential
    libfreetype6-dev
    libpng-dev
    libzmq3-dev
    libspatialindex-dev
    libgl1-mesa-glx
    # gdal-bin
    # libgdal-dev
    python3-gdal
    libsm6
    vim
    wget
    zip

)

apt-get update --fix-missing && apt-get install -y --no-install-recommends ${package_list[@]}
apt-get clean && rm -rf /var/lib/apt/lists/*

# # Install qgis and dependencies
# apt-get update \
#     && apt-get install -y --no-install-recommends wget \
#     && apt-get install -y --no-install-recommends gnupg software-properties-common

apt-get update
# apt install -y gnupg software-properties-common
# wget -qO - https://qgis.org/downloads/qgis-2021.gpg.key | gpg --no-default-keyring --keyring gnupg-ring:/etc/apt/trusted.gpg.d/qgis-archive.gpg --import
# chmod a+r /etc/apt/trusted.gpg.d/qgis-archive.gpg
# add-apt-repository "deb https://qgis.org/ubuntu $(lsb_release -c -s) main"

# apt update
# apt install -y qgis qgis-plugin-grass   

# Install qgis specific gdal dep again
# qgis installation is not proper and not includes gdal_calc and etc.
# specific_qgisgdal_list=(
    
#     gdal-bin=2.2.3+dfsg-2
#     libgdal20=2.2.3+dfsg-2
#     libgdal-dev=2.2.3+dfsg-2
#     libgdal-java=2.2.3+dfsg-2
#     gdal-data=2.2.3+dfsg-2
#     python-gdal=2.2.3+dfsg-2
#     python3-gdal=2.2.3+dfsg-2

# )

apt-get install -y --no-install-recommends ${specific_qgisgdal_list[@]}