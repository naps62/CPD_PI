#!/bin/bash

declare -a CASES

CASES=( tiny small medium big huge )
#COUNTERS=( cpi mem flops l1 l2dca ipb mbaml mbadv )
COUNTERS=( mem )

for c in ${CASES[@]};
do
	for counter in ${COUNTERS[@]};
	do
		timestamp=`date "+%y-%m-%d_%H:%M:%S"`
		echo "${c} - ${counter}";
		time bin/polu.openmp.papi.${counter} data/xml/${c}.param.xml > "results/${c}.${counter}.${timestamp}.hammer" 2> "results/${c}.time.${timestamp}.hammer";
	done;
done;


