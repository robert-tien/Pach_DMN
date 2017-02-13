#!/bin/bash
if [[ -z $1 ]]; then
    echo ""
    echo "missing 1st argument"
    echo "please specify a category, e.g. Car"
    echo "argument 2 is the max_allowed_input length, default is in dmn_plus.py config()"
    echo "argument 3 is the repeat number, default is 2"
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
Yes="Yes"
No="No"
Maybe="Maybe"
Train="Train"
Source="Source"

# python pach_gen_traincases.py -c keyWordPrases/CarMaybe.txt -i dmnNoCarTrain -o genMaybeCar -r 2 -p genMbe
echo "python pach_gen_traincases.py -m $maxinp -c keyWordPrases/$1$Maybe.txt -i dmn$Maybe$1$Source -o gen$Maybe$1 -r $repeat -p genMbe"
python pach_gen_traincases.py -m $maxinp -c keyWordPrases/$1$Maybe.txt -i dmn$Maybe$1$Source -o gen$Maybe$1 -r $repeat -p genMbe
cd /home/robert_tien/work/pachira/dmn/Pach_DMN/data
pwd
#cp gen$Maybe$1/* dmn$No$1$Train
ls gen$Maybe$1/* |wc 
cd ..
fi
