#!/bin/sh

PROF_CONFIG=profile/config
LOG=profile/log.default

# Cuda env vars for profiling
export CUDA_PROFILE=1
export CUDA_PROFLE_CSV=1
export CUDA_PROFILE_CONFIG=$PROF_CONFIG


function logger {
	echo $1 >> $LOG
	echo $1
}

# runs profiling with a given input name 3 times
function profile {
	# get params
	NAME=$1

	for i in 1 2 3; do
		# set env and create dir
		DIR=profile/$NAME.$i
		mkdir -p $DIR
		export CUDA_PROFILE_LOG=$DIR/profile.log
	
		# set logger for this test
		LOG=$DIR/log
		logger "profiling for $NAME$.i started at `date`"
	
		# get input
		INPUT_DIR=input_samples/$NAME
		logsave "copying input from $INPUT_DIR"
		cp -f $INPUT_DIR/foz.xml $INPUT_DIR/velocity.xml $INPUT_DIR/potential.xml .
	
		# running polu.cuda
		./polu.cuda
		mv FVLib.prof FVLib.out FVLib.err
	done
}


# remake the whole project (just to make sure)
make clean && make

# clean current log files (to avoid confusions)
rm FVLib.log FVLib.err FVLib.prof

# run profiler
profile "foz.tiny"
