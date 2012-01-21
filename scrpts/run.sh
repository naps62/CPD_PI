#!/bin/sh

timestamp=`date "+%m-%d_%H:%M:%S"`

resultdir="../results"

cd "bin";

for arg in $@
do
	case $arg in
	polu*)
		case $arg in
		*.cpi|
		*.mem|
		*.flops|
		*.l1)
			outfile = "${resultdir}/${arg}.log"
			;;
		esac
		if [ ! -d "$resultdir" ];
		then
			mkdir "$resultdir";
		fi;
		if [ $outfile ];
		then
			if [ -f "$outfile" ];
			then
				echo >> "$outfile";
			else
				touch "$outfile";
			fi;
			echo "<<==::  ${timestamp}  ::==>>" >> "$outfile";
			./$arg "../data/xml/param.xml" >> "$outfile";
		else
			./$arg "../data/xml/param.xml";
		fi;
		mv "../data/xml/polution.xml" "${resultdir}/${arg}.out.${timestamp}"
		;;
	*)
		./$arg
		;;
	esac
done
