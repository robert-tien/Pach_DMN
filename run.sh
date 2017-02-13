#!/bin/bash
# arg $1 is the class name such HasIns, Car, etc.
if [[ -z $1 ]]; then
    echo ""
    echo "missing argument"
    echo "please specify a category, e.g. Car"
    echo ""
#    exit 1
else 
uuid=$(uuidgen)
echo $uuid
file=$(grep -l -s $uuid *)
echo output file is $file
./util/cleanAll.sh $1 
echo "---- ls data/dmnYes$1Train"
ls -C data/dmnYes$1Train
echo "---- ls data/dmnNo$1Train"
ls -C data/dmnNo$1Train
echo "---- ls data/dmnMaybe$1Source"
ls -C data/dmnMaybe$1Source
#cat HasInsMaybe.txt HasInsNo.txt HasInsYes.txt
echo "---- cat data/keyWordPrases/$1Yes.txt"
cat data/keyWordPrases/$1Yes.txt
echo "---- cat data/keyWordPrases/$1No.txt"
cat data/keyWordPrases/$1No.txt
echo "---- cat data/keyWordPrases/$1Maybe.txt"
cat data/keyWordPrases/$1Maybe.txt
echo "---- ./util/genAll.sh $1 5 2 3 50"
./util/genAll.sh $1 5 2 3 50
echo "---- ./util/distAll.sh $1"
./util/distAll.sh $1
#echo "python dmn_train.py -b p2 -v 150 -m 50 -V 50"
#python dmn_train.py -b p2 -v 150 -m 50 -V 50
echo "---- train: python dmn_train.py -b p2 -v 150 -m 50 -V 50 -r weights_bak/weights_p2_I50S20V150Vb50G523-20KNew"
python dmn_train.py -b p2 -v 150 -m 50 -V 50 -r weights_bak/weights_p2_I50S20V150Vb50G523-20KNew
echo "---- test: python dmn_test.py -b p2 -v 150 -m 50 -V 50 "
python dmn_test.py -b p2 -v 150 -m 50 -V 50  
echo "---- eval taikang HasIns 50 cases"
evalTaikangHasIns.sh
# now grep all useful data 
getDataResults.sh $file
fi
