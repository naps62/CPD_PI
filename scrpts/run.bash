#!/bin/bash

declare -a command;

timestamp=`date "+%Y-%m-%d_%H:%M:%S"`
result_d="results"

#	parse arguments
i=0;
o_flg=false;
for arg;
do
	i=$(( $i + 1 ));
	if [ -z "$exe" ];
	then
		exe="$arg";
	fi;
	case $arg in
	"-o")
		o_flg=true;
		;;
	"-n")
		n_flg=true;
		;;
	*)
		if $o_flg;
		then
			output="$arg";
			o_flg=false;
		elif $n_flg;
		then
			runs="$arg";
			n_flg=false;
		else
			command[$i]="$arg";
		fi;
		;;
	esac;
done;

#	output file
if [ -z "$output" ];
then
	output="${result_d}/${exe}_${timestamp}.log";
fi;

#	number of runs
if [ -z "$runs" ];
then
	runs=1;
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


for i in 1 .. $runs;
do
	date "+%Y-%m-%d_%H:%M:%S" >> $output;
	$command >> $output;
	echo >> "$output";
done;
