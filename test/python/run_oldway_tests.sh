#!/bin/bash

for file in *oldway.py
do
	echo "------------ TEST $file ------------"

	../../build_pytriqs_new $file

	echo
	echo
done

exit 0
