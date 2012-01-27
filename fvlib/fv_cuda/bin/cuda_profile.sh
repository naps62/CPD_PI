#!/bin/sh

PROF_DIR=profile
PROF_CONFIG=profile/config
LOG=profile/log

# Cuda env vars for profiling
export CUDA_PROFILE=1
export CUDA_PROFILE_LOG=$PROF_DIR
export CUDA_PROFLE_CSV=1
export CUDA_PROFILE_CONFIG=$PROF_CONFIG

echo -e "\n\nStarted log at" `date` >> $LOG
