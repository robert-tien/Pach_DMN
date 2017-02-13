#!/bin/bash
# clean dmnYes$1Train/gen*, dmnNo$1Train/gen* and dmnYes$1Test/gen*, dmnNo$1Test/gen* and dmnYes$1Eval/gen*, dmnNo$1Eval/gen*
# clean gen$1Yes, gen$1No, gen$1Maybe directories

if [[ -z $1 ]]; then
    echo ""
    echo "missing argument"
    echo "please specify a category, e.g. Car"
    echo ""
#    exit 1
else 
Train="Train"
Test="Test"
Eval="Eval"
Yes="Yes"
No="No"
Maybe="Maybe"
cd /home/robert_tien/work/pachira/dmn/Pach_DMN/data
pwd
echo "ls dmnYes$1$Train/|wc"
ls dmnYes$1$Train/|wc
echo "ls dmnYes$1$Train/gen*|wc"
ls dmnYes$1$Train/gen*|wc 
echo "ls dmnNo$1$Train/*|wc" 
ls dmnNo$1$Train/*|wc 
echo "ls dmnNo$1$Train/gen*|wc" 
ls dmnNo$1$Train/gen*|wc 
echo "ls dmnYes$1$Test/|wc"
ls dmnYes$1$Test/|wc
echo "ls dmnYes$1$Test/gen*|wc"
ls dmnYes$1$Test/gen*|wc
echo "ls dmnNo$1$Test/|wc"
ls dmnNo$1$Test/|wc
echo "ls dmnNo$1$Test/gen*|wc"
ls dmnNo$1$Test/gen*|wc
echo "ls dmnYes$1$Eval/|wc"
ls dmnYes$1$Eval/|wc
echo "ls dmnYes$1$Eval/gen*|wc"
ls dmnYes$1$Eval/gen*|wc
echo "ls dmnNo$1$Eval/|wc"
ls dmnNo$1$Eval/|wc
echo "ls dmnNo$1$Eval/gen*|wc"
ls dmnNo$1$Eval/gen*|wc
echo "ls gen$Yes$1/*|wc"
ls gen$Yes$1/*|wc
echo "ls gen$No$1/*|wc"
ls gen$No$1/*|wc
echo "ls gen$Maybe$1/*|wc" 
ls gen$Maybe$1/*|wc 
echo "rm dmnYes$1$Train/gen* dmnNo$1$Train/gen* dmnYes$1$Test/gen* dmnNo$1$Test/gen* dmnYes$1$Eval/gen* dmnNo$1$Eval/gen*|wc"
rm -v dmnYes$1$Train/gen* dmnNo$1$Train/gen* dmnYes$1$Test/gen* dmnNo$1$Test/gen* dmnYes$1$Eval/gen* dmnNo$1$Eval/gen* |wc
echo "rm -v gen$Yes$1/* gen$No$1/* gen$Maybe$1/*|wc" 
rm -v gen$Yes$1/* gen$No$1/* gen$Maybe$1/*|wc
echo "mv dmnYes$1$Test/* dmnYes$1$Train"
mv -v dmnYes$1$Test/* dmnYes$1$Train
echo "mv dmnYes$1$Eval/* dmnYes$1$Train"
mv -v dmnYes$1$Eval/* dmnYes$1$Train
echo "mv dmnNo$1$Test/* dmnNo$1$Train"
mv -v dmnNo$1$Test/* dmnNo$1$Train
echo "mv dmnNo$1$Eval/* dmnNo$1$Train"
mv -v dmnNo$1$Eval/* dmnNo$1$Train
echo "ls dmnYes$1$Train|wc"
ls dmnYes$1$Train|wc
echo "ls dmnNo$1$Train|wc"
ls dmnNo$1$Train|wc
cd ..
fi
