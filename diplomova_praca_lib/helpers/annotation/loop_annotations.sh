#!/bin/bash

while read line
do 
	./run_annotate_images_on_gpulab.sh $line < /dev/null
done
