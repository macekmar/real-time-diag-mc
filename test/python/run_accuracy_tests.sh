#!/bin/bash

echo "------------ RUN density.py ------------"
mpirun -np 4 ../../build_pytriqs density.py
echo
echo

echo "------------ RUN impurity.py ------------"
mpirun -np 4 ../../build_pytriqs impurity.py
echo
echo

echo "------------ RUN impurity_oldway.py ------------"
mpirun -np 4 ../../build_pytriqs impurity_oldway.py
echo
echo

../../build_pytriqs compare_with_oldway.py

exit 0
