# -*- coding: utf-8 -*-
import sys
import pdb
import os as os
import numpy as np
import logging
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--pach_out_dir", help="specify pach output dir in 'data' dir")
parser.add_argument("-i", "--pach_in_dir", help="specify pach input dir in 'data' dir")
parser.add_argument("-f", "--pach_offset", type=int, help="specify cut line offset between two iteration. default is 0")
parser.add_argument("-l", "--pach_limit", type=int, help="specify max number of lines of output files default is 40")
#parser.add_argument("-s", "--pach_sent_len", type=int, help="specify max sentence length, default is 20")
args = parser.parse_args()

#pdb.set_trace()
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
if args.pach_offset is not None:
    offset  = args.pach_offset
else: 
    offset = 0
limit = args.pach_limit if args.pach_limit is not None else 40
#sent_len = args.sent_len if args.sent_len is not None else 20
print "max sentence length is specified in config.max_allowed_sent_len"

def output(out_dir, fname, out):
    with open(("/home/robert_tien/work/pachira/dmn/Pach_DMN/data/"+out_dir+"/"+fname),"w") as f:
        for o in out:
            f.write(o)
    return

from dmn_plus import Config
config = Config()
path = "/home/robert_tien/work/pachira/dmn/Pach_DMN/data/"
dir = path+"/"+in_dir
limit = config.max_allowed_inputs 

from pach_input import repacklines

for i, filename in enumerate(os.listdir(dir)):
    fname = dir+"/"+filename
    f = open(fname,"r")
    fn = f.readline() # skip first line - meta info
    if not fn.startswith("FILE"):
        f.close
        continue #generated files
    s = f.read()
    f.close()
    lines = repacklines(s, config, fname)
    if len(lines) < limit:
        continue
    # command to move away long files to be replaced with cut files.
    print "mv "+fname+" tmp"
    j = 0
    for oset in [0, offset]:
        if j ==1 and offset == 0: 
            break
        out = []
        cnt = 0
        for ii, line in enumerate(lines):
            if (oset == 0 and ii%limit == 0 and len(out) > 0) or (oset > 0 and ii%(cnt*limit+oset)==0 and len(out) > 0):
                output(out_dir, (fname[fname.rindex("/"):len(fname)])+".v"+str(j)+"-"+str(cnt),out)
                out = []
                cnt += 1
            out.append(line+"\n") 
        if len(out) > 0:
            output(out_dir, (fname[fname.rindex("/"):len(fname)])+".v"+str(j)+"-"+str(cnt),out)
        j += 1

