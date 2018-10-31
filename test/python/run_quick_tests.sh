#!/bin/bash

for file in run_tests.py phsym_kernels.py
do
    echo "------------ TEST $file 1 proc ------------"

    ../../build_pytriqs $file

    echo
    echo
    echo "------------ TEST $file 4 proc ------------"

    mpirun -np 4 ../../build_pytriqs $file

    echo
    echo
done

for file in collecting.py
do
    echo "------------ TEST $file 4 proc ------------"

    mpirun -np 4 ../../build_pytriqs $file

    echo
    echo
done
exit 0
