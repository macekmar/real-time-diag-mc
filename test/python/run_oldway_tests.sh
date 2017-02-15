#!/bin/bash

for file in *oldway.py
do
	echo "------------ TEST $file ------------"

	../../build_pytriqs $file

	echo
	echo
done

exit 0
