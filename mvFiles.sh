#!/bin/bash
set -x
workdir="/home/robert_tien/work/pachira/dmn/Pach_DMN"
datadir="$workdir/data"
echo $workdir
echo $datadir
if [[ -z $3 ]]; then
    echo "usage: "
    echo "arg $1 is the run log file containing Yes-No, No-Yes, etc"
    echo "arg $2 is either Yes- or No- to specify files to be selected"
    echo "arg $3 is the source directory of each file"
    echo "arg $4 is the destination directory where the files will be moved."
else 
#grep Yes- HasInsFull-I50S20V150Vb50Gen523-20K0Scratch.out | cut -d ' ' -f 4|cut -d '/' -f 10 > stageYesFiles
echo "grep $2 $1 | cut -d ' ' -f 4|cut -d '/' -f 10 > stage$2Files"
grep $2 $1 | cut -d ' ' -f 4|cut -d '/' -f 10 > stage$2Files
#for i in $(cat stageYesFiles); do cp "$i" tmp; done
echo "cd $datadir/$3"
cd $datadir/$3
for i in $(cat $workdir/stage$2Files); do cp -v "$i" $workdir/$4; done
echo "cd $workdir"
cd $workdir
fi
