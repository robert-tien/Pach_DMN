#!/bin/bash
# input: $1 is the directory to precess such as car1, busy1, reject3 etc.

if [[ -z "$1" ]]; then
    echo "duplicate current dir"
    echo "Usage: dup.sh prefix "
    echo "example: dup.sh 1 "
    exit 1
fi

CUR_DIR=$PWD
pwd
#echo "cd $DATA_DIR/$1"
#cd $DATA_DIR/$1
pwd
FILES=*

for f in $FILES
do
      echo "duplicating $f ..."
      cp $f "dup$1_$f"
done
cd $CUR_DIR
pwd
