#!/bin/bash
#	-n N	minimum N runs
#	-t T	trust degree

declare -a command;
declare -a runs;

timestamp=`date "+%Y-%m-%d_%H:%M:%S"`
result_d="results"

#	parse arguments
i=0;
n_flg=false;
t_flg=false;
for arg;
do
	case $arg in
	"-n")
		n_flg=true;
		;;
	"-t")
		t_flg=true;
		;;
	*)
		if $n_flg;
		then
			RUNS="$arg";
			n_flg=false;
		elif $t_flg;
		then
			TRUST="$arg";
			t_flg=false;
		else
			command[$i]="$arg";
			i=$(( $i + 1 ));
		fi;
		;;
	esac;
done;

#	number of runs
if [ -z "$RUNS" ];
then
	RUNS=3;
fi;

#	trust degree
if [ -z "$TRUST" ];
then
	TRUST=5;
fi;

#	create results directory
if [ ! -d "$result_d" ];
then
	mkdir "$result_d";
fi;

#	prepare output file
if [ -f "$output" ];
then
	touch "$output";
fi;

#	main test loop
#	flag	holds the condition value
#	runs	best runs vector
#	rc		runs vector length (count)
#	RUNS	minimal number of runs
#	run		current run
#	rns		time in nano-seconds of the current run
#	rnsi	time in nano-seconds of the iterated run
#	TRUST	trust degree
flag=false
rc=${#runs[@]}
while ! $flag;
do
	run=`${command[@]}`;
	rns=`echo "$run" | grep -o "time:[0-9]\+" | grep -o "[0-9]\+"`;
	for (( i = 0 ; i < $rc ; ++i ));
	do
		rnsi=`echo "${runs[$i]}" | grep -o "time:[0-9]\+" | grep -o "[0-9]\+"`;
		if [ "$rns" -lt "$rnsi" ];
		then
			runi=${runs[$i]};
			rns=$rnsi;
			runs[$i]=$run;
			run=$runi;
		fi;
	done;
	if [ "$rc" -ne "$RUNS" ];
	then
		runs[$rc]=$run;
		rc=${#runs[@]};
	fi;
	if [ "$rc" -eq "$RUNS" ];
	then
		rns=`echo "${runs[0]}" | grep -o "ns:[0-9]\+" | grep -o "[0-9]\+"`;
		flag=true;
		for (( i = 1 ; i < $rc ; ++i ));
		do
			rnsi=`echo "${runs[$i]}" | grep -o "ns:[0-9]\+" | grep -o "[0-9]\+"`;
			diff=$(( $rnsi - $rns ));
			frac=`echo "100*$diff/$rns" | bc`;
			if [ "$frac" -gt "$TRUST" ];
			then
				flag=false;
				break;
			fi;
		done;
	fi;
done;
echo "${runs[0]}";
