# -*- coding: utf-8 -*-
import sys
import pdb
import os as os
import numpy as np
import logging
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--pach_case_file", help="specify pach case file in 'data' dir")
parser.add_argument("-o", "--pach_out_dir", help="specify pach output dir in 'data' dir")
parser.add_argument("-i", "--pach_in_dir", help="specify pach input dir in 'data' dir")
parser.add_argument("-r", "--pach_repeat", default=1, help="specify repeat number for each file")
parser.add_argument("-p", "--pach_prefix", default="gen", help="specify generated file prefix")
parser.add_argument("-m", "--max_inputs", type=int, help="specify max allowed input length")
args = parser.parse_args()

from dmn_plus import Config
config = Config()

if args.pach_case_file is not None:
    casefile = args.pach_case_file
else: 
    print "no case file"
    sys.exit(1)
if args.pach_out_dir   is not None:
    out_dir  = args.pach_out_dir
else: 
    print "no out_dir"
    sys.exit(1)
if args.pach_in_dir   is not None:
    in_dir  = args.pach_in_dir
else: 
    print "no in_dir"
    sys.exit(1)
if args.pach_repeat   is not None:
    repeat  = int(args.pach_repeat)
if args.pach_prefix   is not None:
    prefix  = args.pach_prefix
config.max_allowed_inputs = args.max_inputs if args.max_inputs is not 0 else config.max_allowed_inputs

def output(out_dir, fname, out):
    found = 0
    for o in out:
        if o.find("保险") > 0:
            found = 1
            break
    if found == 0:
        #pdb.set_trace()
        return # residual file fragment not containing 
    with open(("/home/robert_tien/work/pachira/dmn/Pach_DMN/data/"+out_dir+"/"+fname),"w") as f:
        for o in out:
            f.write(o)
    return
#pdb.set_trace()

path = "/home/robert_tien/work/pachira/dmn/Pach_DMN/data/"
# read case snippet files
with open((path+casefile)) as f:
    case_list = []
    case = []
    for line in f:
        if line.startswith("#"):
            continue
        if line.startswith("---"):
            case_list.append(case)
            case = []
            continue
        case.append(line)

nr_cases = len(case_list)
#max_case_len = np.max([len(case) for case in case_list])
margin = 10 # to accomodate inserted case lines
limit = config.max_allowed_inputs-margin 
# insert into existing dmnNoHasInsTrain/ dir cases randomly
dir = path+in_dir
nr_samples= len(os.listdir(dir))
# random permute cases, assume max number of packed lines < 1000
p = np.random.permutation(nr_samples*repeat+1000)

from pach_input import repacklines

#pdb.set_trace()
for i, filename in enumerate(os.listdir(dir)):
    fname = dir+"/"+filename
    f = open(fname,"r")
    fn = f.readline() # skip first line - meta info
    s = f.read()
    f.close()
    lines = repacklines(s, config, fname)
    for j in range(repeat):
        case = out = []
        insert_pt = 0
        for ii, line in enumerate(lines):
            if ii%limit== 0 :
                case=case_list[p[i*(j+1)+ii]%nr_cases] # randomly pick a case
                if len(case) == 0: # or ii == 80:
                    pdb.set_trace()
                insert_pt = random.randint(0,min(len(lines)-1,limit-1))
                if len(out) > 0 :
                    output(out_dir, prefix+str(i)+"-"+str(ii)+"-"+str(j), out)
                out = []
            out.append(line+"\n")
            if ii%limit == insert_pt:
                out.append("\n")
                for l in case:
                    out.append(l)
                out.append("\n")

        if ii%limit > 0 and ii%limit < insert_pt:
            out.append("\n")
            for l in case:
                out.append(l)
            out.append("\n")

        if len(out) > 0 :
            output(out_dir, prefix+str(i)+"-"+str(ii)+"-"+str(j), out)

