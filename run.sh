#!/bin/bash
# arg $1 is the class name such HasIns, Car, etc.
if [[ -z $1 ]]; then
    echo ""
    echo "missing argument"
    echo "please specify a category, e.g. Car"
    echo ""
#    exit 1
else 
echo "<html>" 
<head>
<meta http-equiv="content-type" content="text/html;charset=utf-8">
</head>
#echo "<a name=\"\"></a>"
echo "<a href=\"#cleanAll\">cleanAll</a>"
echo "<a href=\"#dmnYes$1Train\">dmnYes$1Train</a>"
echo "<a href=\"#dmnNo$1Train\">dmnNo$1Train</a>"
echo "<a href=\"#dmnMaybe$1Source\">dmnMaybe$1Source</a>"
echo "<a href=\"#keyWordPrases/$1Yes.txt\">keyWordPrases/$1Yes.txt</a>"
echo "<a href=\"#keyWordPrases/$1No.txt\">keyWordPrases/$1No.txt</a>"
echo "<a href=\"#keyWordPrases/$1Maybe.txt\">keyWordPrases/$1Maybe.txt</a>"
echo "<a href=\"#genAll\">genAll</a>"
echo "<a href=\"#distAll\">distAll</a>"
echo "<a href=\"#train\">train</a>"
echo "<a href=\"#test\">test</a>"
echo "<a href=\"#eval taikang\">eval taikang</a>"
echo "<a href=\"#attention.dmp\">attention.dmp</a>"
echo "<a name=\"cleanAll\"></a>"
# get the output file name
uuid=$(uuidgen)
echo $uuid
file=$(grep -l -s $uuid *)
echo output file is $file
./util/cleanAll.sh $1 
echo "<a name=\"dmnYes$1Train\"></a>"
echo "---- ls data/dmnYes$1Train"
ls -C data/dmnYes$1Train
echo "<a name=\"dmnNo$1Train\"></a>"
echo "---- ls data/dmnNo$1Train"
ls -C data/dmnNo$1Train
echo "<a name=\"dmnMaybe$1Source\"></a>"
echo "---- ls data/dmnMaybe$1Source"
ls -C data/dmnMaybe$1Source
#cat HasInsMaybe.txt HasInsNo.txt HasInsYes.txt
echo "<a name=\"keyWordPrases/$1Yes.txt\"></a>"
echo "---- cat data/keyWordPrases/$1Yes.txt"
cat data/keyWordPrases/$1Yes.txt
echo "<a name=\"keyWordPrases/$1No.txt\"></a>"
echo "---- cat data/keyWordPrases/$1No.txt"
cat data/keyWordPrases/$1No.txt
echo "<a name=\"keyWordPrases/$1Maybe.txt\"></a>"
echo "---- cat data/keyWordPrases/$1Maybe.txt"
cat data/keyWordPrases/$1Maybe.txt
echo "<a name=\"genAll\"></a>"
echo "---- ./util/genAll.sh $1 5 2 3 50"
./util/genAll.sh $1 5 2 3 50
echo "<a name=\"distAll\"></a>"
echo "---- ./util/distAll.sh $1"
./util/distAll.sh $1
echo "<a name=\"train\"></a>"
echo "python dmn_train.py -b p2 -v 150 -m 50 -V 50"
python dmn_train.py -b p2 -v 150 -m 50 -V 50
#echo "---- train: python dmn_train.py -b p2 -v 150 -m 50 -V 50 -r weights_bak/weights_p2_I50S20V150Vb50G523-20KNew"
#python dmn_train.py -b p2 -v 150 -m 50 -V 50 -r weights_bak/weights_p2_I50S20V150Vb50G523-20KNew
echo "<a name=\"test\"></a>"
echo "---- test: python dmn_test.py -b p2 -v 150 -m 50 -V 50 "
python dmn_test.py -b p2 -v 150 -m 50 -V 50  
echo "<a name=\"eval taikang\"></a>"
echo "---- eval taikang HasIns 50 cases"
evalTaikangHasIns.sh
# now grep all useful data 
echo "generate result html"
getDataResults.sh $file > run_tmp.out
createHtml.sh run_tmp.out
echo "</html>" 
fi
