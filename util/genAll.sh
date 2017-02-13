#!/bin/bash
if [[ -z $1 ]]; then
    echo ""
    echo "missing argument"
    echo "please specify a category, e.g. Car"
    echo "argument 2 is the repeat number for genYes, default is 3"
    echo "argument 3 is for No, default is 2"
    echo "argument 4 is for Maybe, default is 2"
    echo "argument 5 is for maxed_allowed_input length, default is in dmn_plus.py config()"
    echo ""
#    exit 1
else 
Train="Train"
cd /home/robert_tien/work/pachira/dmn/Pach_DMN/
source util/genYes.sh $1 $2 $5
source util/genNo.sh $1 $3 $5
source util/genMaybe.sh $1 $4 $5
cd data
echo "cp genYes$1/* dmnYes$1$Train"
cp genYes$1/* dmnYes$1$Train
echo "ls dmnYes$1$Train|wc"
ls dmnYes$1$Train|wc
echo "cp genNo$1/* dmnNo$1$Train"
cp genNo$1/* dmnNo$1$Train
echo "cp genMaybe$1/* dmnNo$1$Train"
cp genMaybe$1/* dmnNo$1$Train
echo "ls dmnNo$1$Train|wc"
ls dmnNo$1$Train|wc
cd ..
fi
