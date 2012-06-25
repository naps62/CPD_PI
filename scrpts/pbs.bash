#!/bin/bash
#
#PBS -l walltime=3:00:00
#PBS -M pdrcosta90@gmail.com
#PBS -m bea
#

if [ ! "$RUNS" ]; then RUNS="1"; fi;

cd "$PBS_O_WORKDIR";

R="1";
while [ "$R" -le "$RUNS" ];
do
	echo -en "\t${R}..." 1>&2;
	if [ "$THREADS" ];
	then
		$EXEC "data/xml/huge.param.xml" "$THREADS";
	else
		echo "$PWD";
		$EXEC "data/xml/huge.param.xml";
	fi;
	echo "DONE" 1>&2;
	R=$(( $R + 1 ));
done;
