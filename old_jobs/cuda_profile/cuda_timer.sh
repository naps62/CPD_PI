#!/bin/sh
#PBS -N cuda_profiler
#PBS -l nodes=1:tesla:ppn=24
#PBS -l walltime=5:00:00
#PBS -m bea
#PBS -e output/cuda_profile.err
#PBS -o output/cuda_profile.out

# goto working dir
WORK_DIR=~/projects/pi/fvlib/fv_cuda/bin
cd $WORK_DIR

PROF_CONFIG=profile_timer/config
LOG=profile_timer/log

function logger {
	echo `date` $1 >> $LOG
	echo `date` $1
}

function profile_single {
	# get params
	NAME=$1
	CONFIG=$2

	DIR=profile_timer/$NAME
	mkdir -p $DIR
	
	# if no config is set, run without PROFILE, to avoid overhead
	# and count event timers
	if [ -z $CONFIG ]; then
		export CUDA_PROFILE=0
		rm -f FVLib.prof FVLib.log FVLib.err
		logger "profiling $NAME for FVLib.prof"
		./polu.cuda
		mv FVLib.prof FVLib.log FVLib.err $DIR
	else
		export CUDA_PROFILE=1
		export CUDA_PROFILE_CSV=1
		export CUDA_PROFILE_CONFIG=$PROF_CONFIG/$CONFIG
		export CUDA_PROFILE_LOG=$DIR/profile.$CONFIG
		logger "profiling $NAME for profile.$CONFIG"
		./polu.cuda
	fi
	
}

# runs profiling with a given input name
# first run is without profile, to use FVLib.prof without the possible profiling overhead
# after that, for each config.* file present, 3 runs are made
function profile {
	# get params
	NAME=$1

	logger "starting profile for $NAME"
	rm -r profile/$NAME

	# get input
	INPUT_DIR=input_samples/$NAME
	logger "copying input from $INPUT_DIR"
	ln -sf $INPUT_DIR/foz.xml 				foz.xml
	ln -sf $INPUT_DIR/velocity.xml			velocity.xml
	ln -sf $INPUT_DIR/potential.xml			potential.xml
	ln -sf $INPUT_DIR/concentration_ini.xml	concentration_ini.xml
	#cp -f $INPUT_DIR/foz.xml $INPUT_DIR/velocity.xml $INPUT_DIR/potential.xml $INPUT_DIR/concentration_ini.xml .

	profile_single $NAME
	#for config in `ls profile/config`; do
	#	profile_single $NAME $config
	#done

	logger "done profiling $NAME"
}


# remake the whole project (just to make sure)
# cd .. && make clean && make && cd -

# clean current log files (to avoid confusions)
rm -f FVLib.log FVLib.err FVLib.prof

# run profiler
profile "foz.tiny"
profile "foz.small"
profile "foz.medium"
profile "foz.big"
profile "foz.huge"
