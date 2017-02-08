#!/bin/bash

for file in *.py
do
	echo "------------ TEST $file ------------"

	../../build_pytriqs_new $file

	echo
	echo
done

exit 0
