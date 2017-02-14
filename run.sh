#!/bin/bash
# arg $1 is the class name such HasIns, Car, etc.
if [[ -z $1 ]]; then
    echo ""
    echo "missing argument"
    echo "please specify a category, e.g. Car"
    echo "to improve on a previous run, enter previous run log file"
    echo ""
#    exit 1
else 
echo "<html>" 
echo "<head>"
echo "<meta http-equiv=\"content-type\" content=\"text/html;charset=utf-8\">"
echo "</head>"
#echo "<a name=\"\"></a>"
echo "<a href=\"#improve\">improve </a> <br>"
echo "<a href=\"#cleanAll\">cleanAll</a> <br>"
echo "<a href=\"#dmnYes$1Train\">dmnYes$1Train</a> <br>"
echo "<a href=\"#dmnNo$1Train\">dmnNo$1Train</a> <br>"
echo "<a href=\"#dmnMaybe$1Source\">dmnMaybe$1Source</a> <br>"
echo "<a href=\"#keyWordPrases/$1Yes.txt\">keyWordPrases/$1Yes.txt</a> <br>"
echo "<a href=\"#keyWordPrases/$1No.txt\">keyWordPrases/$1No.txt</a> <br>"
echo "<a href=\"#keyWordPrases/$1Maybe.txt\">keyWordPrases/$1Maybe.txt</a> <br>"
echo "<a href=\"#genAll\">genAll</a> <br>"
echo "<a href=\"#distAll\">distAll</a> <br>"
echo "<a href=\"#train\">train</a> <br>"
echo "<a href=\"#test\">test</a> <br>"
echo "<a href=\"#eval taikang\">eval taikang</a> <br>"
echo "<a href=\"#attention.dmp\">attention.dmp</a> <br>"
echo "<a name=\"improve\"></a> <br>"
echo "<PRE>"
if [ "$#" -eq "2" ]; then
    improve.sh $2
fi
echo "</PRE>"
echo "<a name=\"cleanAll\"></a> <br>"
echo "<PRE>"
# get the output file name
uuid=$(uuidgen)
echo $uuid
file=$(grep -l -s $uuid *)
echo output file is $file
./util/cleanAll.sh $1 
echo "</PRE>"
echo "<a name=\"dmnYes$1Train\"></a>"
echo "<PRE>"
echo "---- ls data/dmnYes$1Train"
ls -C data/dmnYes$1Train
echo "</PRE>"
echo "<a name=\"dmnNo$1Train\"></a>"
echo "<PRE>"
echo "---- ls data/dmnNo$1Train"
ls -C data/dmnNo$1Train
echo "</PRE>"
echo "<a name=\"dmnMaybe$1Source\"></a>"
echo "<PRE>"
echo "---- ls data/dmnMaybe$1Source"
ls -C data/dmnMaybe$1Source
#cat HasInsMaybe.txt HasInsNo.txt HasInsYes.txt
echo "</PRE>"
echo "<a name=\"keyWordPrases/$1Yes.txt\"></a>"
echo "<PRE>"
echo "---- cat data/keyWordPrases/$1Yes.txt"
cat data/keyWordPrases/$1Yes.txt
echo "</PRE>"
echo "<a name=\"keyWordPrases/$1No.txt\"></a>"
echo "<PRE>"
echo "---- cat data/keyWordPrases/$1No.txt"
cat data/keyWordPrases/$1No.txt
echo "</PRE>"
echo "<a name=\"keyWordPrases/$1Maybe.txt\"></a>"
echo "---- cat data/keyWordPrases/$1Maybe.txt"
cat data/keyWordPrases/$1Maybe.txt
echo "</PRE>"
echo "<a name=\"genAll\"></a>"
echo "<PRE>"
echo "---- ./util/genAll.sh $1 5 2 3 50"
./util/genAll.sh $1 5 2 3 50
echo "</PRE>"
echo "<a name=\"distAll\"></a>"
echo "<PRE>"
echo "---- ./util/distAll.sh $1"
./util/distAll.sh $1
echo "</PRE>"
echo "<a name=\"train\"></a>"
echo "<PRE>"
#echo "python dmn_train.py -b p2 -v 150 -m 50 -V 50"
#python dmn_train.py -b p2 -v 150 -m 50 -V 50
echo "---- train: python dmn_train.py -b p2 -v 150 -m 50 -V 50 -r weights_bak/weights.HasInsFull-I50S20V150Vb50Gen523-20K0Scratch"
python dmn_train.py -b p2 -v 150 -m 50 -V 50 -r weights_bak/weights.HasInsFull-I50S20V150Vb50Gen523-20K0Scratch
echo "</PRE>"
echo "<a name=\"test\"></a>"
echo "<PRE>"
echo "---- test: python dmn_test.py -b p2 -v 150 -m 50 -V 50 "
python dmn_test.py -b p2 -v 150 -m 50 -V 50  
echo "</PRE>"
echo "<a name=\"eval taikang\"></a>"
echo "<PRE>"
echo "---- eval taikang HasIns 50 cases"
evalTaikangHasIns.sh
#echo "</PRE>" inside evalTaikangHasIns.sh
# now grep all useful data 
echo "generate result html"
#getDataResults.sh $file > run_tmp.out
createHtml.sh $file
echo "</html>" 
fi
