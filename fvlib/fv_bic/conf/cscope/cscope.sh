#!/bin/bash

EXT_ALLOW=conf/cscope/allow.ext
FILE_DENY=conf/cscope/deny.files

TMP_FILE=cscope.files.tmp

rm -f $TMP_FILE
touch $TMP_FILE

for ext in `cat $EXT_ALLOW`; do
	find . -name "$ext" >> $TMP_FILE
done

#for deny in $FILE_DENY; do
	grep -Evf $FILE_DENY $TMP_FILE > cscope.files
#done

#rm $TMP_FILE
