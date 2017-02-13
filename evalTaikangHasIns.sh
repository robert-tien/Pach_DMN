#!/bin/bash
# arg $1 is the weight directory. if none, then dmn_test.py will default to weights dir.

#set -x
#python dmn_test.py -b p2 -e 1 -d taikang_hasIns -v 150 -V 50 -k "保险" -s 1 -w weights_bak/weights.HasInsTrain-I50S20V150Vb50Gen523-20kK3
#python dmn_test.py -b p2 -e 1 -d taikang_hasIns -v 150 -V 50 -k "保险" -s 1 -w weights_bak/weights.HasInsTrain-I50S20V150Vb50Gen523-20kK3Incr1
#python dmn_test.py -b p2 -v 150 -V 50 -k "保险" -s 1 
echo "mv -i data/dmnYesHasInsTest data/dmnYesHasInsTest_bk"
mv -i data/dmnYesHasInsTest data/dmnYesHasInsTest_bk
echo "mv -i data/dmnNoHasInsTest data/dmnNoHasInsTest_bk"
mv -i data/dmnNoHasInsTest data/dmnNoHasInsTest_bk
echo "mv -i data/evalTaikangYes data/dmnYesHasInsTest"
mv -i data/evalTaikangYes data/dmnYesHasInsTest
echo "mv -i data/evalTaikangNo data/dmnNoHasInsTest"
mv -i data/evalTaikangNo data/dmnNoHasInsTest
if [[ -z $1 ]]; then
echo "python dmn_test.py -b p2 -v 150 -V 50 -k "保险" -s 1"
python dmn_test.py -b p2 -v 150 -V 50 -k "保险" -s 1 
else
#python dmn_test.py -b p2 -v 150 -V 50 -k "保险" -s 1 -w weights_bak/weights.HasInsTrain-I50S20V150Vb50Gen523-20kK2
#python dmn_test.py -b p2  -v 150 -V 50 -k "保险" -s 1 -w weights_bak/weights.HasInsTrain-I50S20V150Vb50Gen523-20kK3
#python dmn_test.py -b p2 -v 150 -V 50 -k "保险" -s 1 -w weights_bak/weights.HasInsTrain-I50S20V150Vb50Gen523-20kK3Incr1
#python dmn_test.py -b p2 -v 150 -V 50 -k "保险" -s 1 -w weights_bak/weights_p2_I50S20V150Vb50G523-20KNew
echo "python dmn_test.py -b p2 -v 150 -V 50 -k "保险" -s 1 -w $1"
python dmn_test.py -b p2 -v 150 -V 50 -k "保险" -s 1 -w $1
fi
echo "mv -i data/dmnYesHasInsTest data/evalTaikangYes"
mv -i data/dmnYesHasInsTest data/evalTaikangYes
echo "mv -i data/dmnNoHasInsTest data/evalTaikangNo"
mv -i data/dmnNoHasInsTest data/evalTaikangNo
echo "mv -i data/dmnYesHasInsTest_bk data/dmnYesHasInsTest"
mv -i data/dmnYesHasInsTest_bk data/dmnYesHasInsTest
echo "mv -i data/dmnNoHasInsTest_bk data/dmnNoHasInsTest"
mv -i data/dmnNoHasInsTest_bk data/dmnNoHasInsTest
echo "#dumping attention.dmp"
echo "-----------------------"
echo "<a name=\"attention.dmp\"></a>"
cat attention.dmp
