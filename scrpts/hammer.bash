#!/bin/bash

declare -a CASES

CASES=( tiny small medium big huge )
COUNTERS=( cpi mem flops l1 l2dca ipb mbaml mbadv )

for c in ${CASES[@]};
do
	for counter in ${COUNTERS[@]};
	do
		timestamp=`date "+%y-%m-%d_%H:%M:%S"`
		echo "${c} - ${counter}";
		bin/polu.clean.papi.${counter} data/xml/${c}.param.xml > "results/${c}.${counter}.${timestamp}.hammer";
	done;
done;


