#!/bin/bash
if [[ -z $1 ]]; then
    echo ""
    echo "missing argument"
    echo "please specify a category, e.g. Car"
    echo "argument 2 is the repeat number, default is 3"
    echo ""
#    exit 1
else 
Train="Train"
Test="Test"
Eval="Eval"
cd /home/robert_tien/work/pachira/dmn/Pach_DMN/
echo "python rand.py -d data/dmnYes$1$Train -o data/dmnYes$1$Test > tmpmp1.sh"
python rand.py -d data/dmnYes$1$Train -o data/dmnYes$1$Test > tmpmp1.sh
mkdir data/dmnYes$1$Test
source tmpmp1.sh
ls data/dmnYes$1$Test |wc
echo "python rand.py -d data/dmnYes$1$Train -o data/dmnYes$1$Eval > tmpmp2.sh"
python rand.py -d data/dmnYes$1$Train -o data/dmnYes$1$Eval > tmpmp2.sh
mkdir data/dmnYes$1$Eval
source tmpmp2.sh
ls data/dmnYes$1$Eval|wc
echo "python rand.py -d data/dmnNo$1$Train -o data/dmnNo$1$Test > tmpmp3.sh"
python rand.py -d data/dmnNo$1$Train -o data/dmnNo$1$Test > tmpmp3.sh
mkdir data/dmnNo$1$Test
source tmpmp3.sh
ls data/dmnNo$1$Test|wc
echo "python rand.py -d data/dmnNo$1$Train -o data/dmnNo$1$Eval > tmpmp4.sh"
python rand.py -d data/dmnNo$1$Train -o data/dmnNo$1$Eval > tmpmp4.sh
mkdir data/dmnNo$1$Eval
source tmpmp4.sh
ls data/dmnNo$1$Eval|wc
echo "ls data/dmnYes$1$Train|wc"
ls data/dmnYes$1$Train|wc
echo "ls data/dmnNo$1$Train|wc"
ls data/dmnNo$1$Train|wc
fi
