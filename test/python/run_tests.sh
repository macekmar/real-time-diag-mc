#!/bin/bash

for file in *.py
do
	if [[ ! $file == *oldway.py ]]
	then
		echo "------------ TEST $file ------------"

		../../build_pytriqs $file

		echo
		echo
	fi
done

exit 0
