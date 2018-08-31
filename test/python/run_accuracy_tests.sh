#!/bin/bash

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
