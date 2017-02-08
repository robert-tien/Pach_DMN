# -*- coding: utf-8 -*-
import argparse
#import numpy as np
import os as os
import pdb
import sys
import random

# this script will random choose 10% of files in input_dir and mv to out_dir

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--input_dir", help="specify input dir")
parser.add_argument("-o", "--out_dir", help="specify output dir")

args = parser.parse_args()
dname = args.input_dir
oname = args.out_dir
#pdb.set_trace()

flist = os.listdir(dname)
#print  os.listdir(dir)

#for fname in os.listdir(args.input_dir):
#    print fname
rflist = []
ii = 0
print "#number of total: "+str(len(flist))
for i in range(1, len(flist)):
    f=random.choice(flist)
    if f not in rflist:
        rflist.append(f)
        ii += 1
    if ii > len(flist)/10:
        break
for fn in rflist:
   print "mv -i "+dname+"/"+fn+" "+oname

