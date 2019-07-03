#!/bin/bash


echo "Testing: quasimc"
rm Test2.hdf5
mpiexec -np 2 python quasimc.py
rm Test2.hdf5
echo "Testing: quasimc_postprocessing.py"
python quasimc_postprocessing.py
echo "Testing: quasimc_model.py"
python quasimc_model.py
echo "Testing: quasimc_coor_transforms.py"
python quasimc_coor_transforms.py
echo "Testing: quasimc_generators.py"
python quasimc_generators.py