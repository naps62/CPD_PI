#!/bin/sh

cd "bin";

for arg in $@
do
	case $arg in
	polu*)
		time ./$arg "../data/xml/param.xml";
		;;
	*)
		time ./$arg
		;;
	esac
done
