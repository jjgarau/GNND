#!/usr/bin/env bash

#Run this file (source install_torch_geometric.sh) instead of installing through a requirements.txt to install dependencies.
#Argument options are cu92, cu101, cu102, cu110. Default is cpu.
if [ $# -eq 0 ]
then
    CUDA="cpu"
else
    CUDA=${1}
    if [ "$CUDA" != "cpu" -a "$CUDA" != "cu92" -a "$CUDA" != "cu101" -a "$CUDA" != "cu102" -a "$CUDA" != "cu110" ]
    then
        echo The given argument is invalid. Options are cpu, cu92, cu101, cu102, and cu110. Default is cpu.
        CUDA="None"
    fi
fi

if [ "$CUDA" != "None" ]
then
#    pip3 install torch
#    pip3 install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.7.0.html
#    pip3 install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.7.0.html
#    pip3 install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.7.0.html
#    pip3 install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.7.0.html
#    pip3 install torch-geometric
#    pip3 install torch-geometric-temporal
#    pip3 install geopy

    # install geos
    brew install geos
    cd /usr/lib
    sudo ln -s libgeos-3.3.3.so libgeos.so
    sudo ln -s libgeos-3.3.3.so libgeos.so.1
    curl -OL https://github.com/matplotlib/basemap/archive/master.zip
    cd /Users/natesesti/Desktop/GNND/GNND
    pip install master.zip
fi