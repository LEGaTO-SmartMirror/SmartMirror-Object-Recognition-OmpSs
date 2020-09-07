#!/bin/bash


cd /opt/dev/MagicMirror/modules/SmartMirror-Object-Recognition-OmpSs/object_detection_ompss/build

source	env.sh

mpirun.mpich -np 2 -f hostfile ./objDetect.bin
