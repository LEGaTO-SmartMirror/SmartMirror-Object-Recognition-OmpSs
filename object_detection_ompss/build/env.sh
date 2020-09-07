#!/bin/bash

NOS6=/opt/dev/ompss@cluster/bin/oss2cluster/nanos6
MCXX=/opt/dev/ompss@cluster/bin/oss2cluster/mcxx
#DARKNET=/opt/dev/ompss@cluster/darknet
DARKNET=/opt/dev/dependencies/darknet
OCV=/usr/local/include/opencv4
VAL=/opt/dev/ompss@cluster/bin/valgrind


export PATH=$MCXX/bin:$PATH
export LD_LIBRARY_PATH=$MCXX/lib:$LD_LIBRARY_PATH
#export LD_RUN_PATH=$MCXX/share/mcxx:$LD_RUN_PATH
export LD_RUN_PATH=$MCXX/bin:$LD_RUN_PATH

export C_INCLUDE_PATH=$NOS6/include:$C_INCLUDE_PATH
export LD_INCLUDE_PATH=$NOS6/include:$LD_INCLUDE_PATH
export C_LIBRARY_PATH=$NOS6/lib:$C_LIBRARY_PATH
export LD_LIBRARY_PATH=$NOS6/lib:$LD_LIBRARY_PATH
export PATH=$NOS6/bin:$PATH

export C_INCLUDE_PATH=$DARKNET/src:$C_INCLUDE_PATH
export LD_INCLUDE_PATH=$DARKNET/src:$LD_INCLUDE_PATH
export C_INCLUDE_PATH=$DARKNET/include:$C_INCLUDE_PATH
export LD_INCLUDE_PATH=$DARKNET/include:$LD_INCLUDE_PATH
export C_LIBRARY_PATH=$DARKNET/:$C_LIBRARY_PATH
export LD_LIBRARY_PATH=$DARKNET/:$LD_LIBRARY_PATH
export CPP_INCLUDE_PATH=$DARKNET/include:$CPP_INCLUDE_PATH

export C_INCLUDE_PATH=$OCV:$C_INCLUDE_PATH
export CPP_INCLUDE_PATH=$OCV:$CPP_INCLUDE_PATH

export PATH=$VAL/bin:$PATH
export VALGRIND_LIB=$VAL/lib/valgrind

#export NANOS6=debug
export NANOS6=optimized
export NANOS6_SCHEDULER=locality
export NANOS6_CLUSTER_SCHEDULING_POLICY=locality
export NANOS6_COMMUNICATION=mpi-2sided
export NANOS6_LOCAL_MEMORY=8GB
#export NANOS6_DISTRIBUTED_MEMORY=1GB
export NANOS6_LOADER_VERBOSE=1


export PATH=/opt/dev/ompss@cluster/bin/tig/bin:$PATH

echo "OmpSs-2 Loaded With The Following ENV-ARGs:"
env | grep NAN
