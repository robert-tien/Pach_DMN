#!/bin/bash
if [[ -z $1 ]]; then
    echo ""
    echo "missing argument"
    echo "please specify a category, e.g. Car"
    echo "argument 2 is the max_allowed_input length, default is in dmn_plus.py config()"
    echo "argument 3 is the repeat number, default is 3"
    echo ""
#    exit 1
else 
    if [[ -z $2 ]]; then
        repeat="3"
    else 
        repeat=$2
    fi
    if [[ -z $3 ]]; then
        maxinp="0"
    else 
        maxinp=$3
    fi
Train="Train"
Yes="Yes"
No="No"
Maybe="Maybe"
Train="Train"
Source="Source"

#python pach_gen_traincases.py -c keyWordPrases/CarYes.txt -i dmnNoCarTrain -o genYesCar -r 3 -p genYes
echo "python pach_gen_traincases.py -m $maxinp -c keyWordPrases/$1$Yes.txt -i dmn$Maybe$1$Source -o gen$Yes$1 -r $repeat -p gen$Yes"
python pach_gen_traincases.py -m $maxinp -c keyWordPrases/$1$Yes.txt -i dmn$Maybe$1$Source -o gen$Yes$1 -r $repeat -p gen$Yes
cd /home/robert_tien/work/pachira/dmn/Pach_DMN/data
pwd
#cp gen$Yes$1/* dmn$Yes$1$Train
echo "ls gen$Yes$1/* |wc "
ls gen$Yes$1/* |wc 
cd ..
fi
