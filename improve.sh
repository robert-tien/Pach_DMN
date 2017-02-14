#!/bin/bash
# arg $1 log file used to improve
if [[ -z $1 ]]; then
    echo "please enter log file"
else 
mvFiles.sh $1 Yes- taikang_hasIns dmnYesHasInsTrain
mvFiles.sh $1 No- taikang_hasIns dmnMaybeHasInsSource
mvFiles.sh $1 No- taikang_hasIns dmnNoHasInsTrain    
echo "cat tmpYes >> data/keyWordPrases/HasInsYes.txt"
echo "cat tmpNo >> data/keyWordPrases/HasInsNo.txt"
echo "cat tmpMbe >> data/keyWordPrases/HasInsMaybe.txt"
cat tmpYes >> data/keyWordPrases/HasInsYes.txt
cat tmpNo >> data/keyWordPrases/HasInsNo.txt
cat tmpMbe >> data/keyWordPrases/HasInsMaybe.txt
fi
