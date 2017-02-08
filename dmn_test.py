# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

import time
import argparse
import pdb

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--pach_task_id", help="specify pach task")
parser.add_argument("-t", "--dmn_type", help="specify type of dmn (default=original)")
parser.add_argument("-s", "--batch_size", help="specify pach task")
parser.add_argument("-d", "--eval_dir", help="specify eval dir in data dir")
parser.add_argument("-e", "--eval_mode", help="specify eval mode")
parser.add_argument("-r", "--repack", help="specify if output repack files, default=False")
parser.add_argument("-v", "--vec_size", type=int, help="specify GloVe vec_size, default=100")
parser.add_argument("-m", "--max_inputs", type=int, help="specify max allowed input length")
parser.add_argument("-V", "--min_vocab_cnt", type=int, help="specify min vocab count")
parser.add_argument("-w", "--weight_dir", help="specify weight dir")
parser.add_argument("-k", "--key_words", help="specify key words")
args = parser.parse_args()

dmn_type = args.dmn_type if args.dmn_type is not None else "plus"

if dmn_type == "original":
    from dmn_original import Config
    config = Config()
elif dmn_type == "plus":
    from dmn_plus import Config
    config = Config()
else:
    raise NotImplementedError(dmn_type + ' DMN type is not currently implemented')

#pdb.set_trace()
if args.pach_task_id is not None:
    config.pach_id = args.pach_task_id
if args.batch_size is not None:
    config.batch_size = int(args.batch_size)
if args.eval_dir  is not None:
    config.eval_dir  = args.eval_dir
if args.eval_mode is not None:
    config.eval_mode = True
if args.repack is not None:
    config.repack = False
config.vec_size = args.vec_size if args.vec_size is not None else 100
config.hidden_size = config.embed_size = config.vec_size
config.max_allowed_inputs = args.max_inputs if args.max_inputs is not None else config.max_allowed_inputs
config.min_vocab_cnt = args.min_vocab_cnt if args.min_vocab_cnt is not None else config.min_vocab_cnt
weight_dir = args.weight_dir if args.weight_dir is not None else "weights"
key_words = args.key_words if args.key_words is not None else ""

config.strong_supervision = False

config.train_mode = False

if config.eval_mode:
    print 'Evaluating DMN ' + dmn_type + ' on pach task', config.pach_id
else:
    print 'Testing DMN ' + dmn_type + ' on pach task', config.pach_id

print "max_input_len="+str(config.max_allowed_inputs)
print "max_allowed_sent_len="+str(config.max_allowed_sent_len)
print "min_vocab_cnt ="+str(config.min_vocab_cnt)
print "episodic memory num_hops="+str(config.num_hops)
print "embed_size ="+str(config.embed_size) # 100
print "hidden_size ="+str(config.hidden_size) # 100

#pdb.set_trace()
def hasKeywords(key_words, fn):
    f = open(fn, "r")
    s = f.read()
    for w in key_words.split():
        if s.find(w) > 0: 
            return 1
    return 0

# create model
with tf.variable_scope('DMN') as scope:
    if dmn_type == "original":
        from dmn_original import DMN
        model = DMN(config)
    elif dmn_type == "plus":
        from dmn_plus import DMN_PLUS
        model = DMN_PLUS(config)

print '==> initializing variables'
init = tf.global_variables_initializer()
#init = tf.initialize_all_variables()
saver = tf.train.Saver()

with tf.Session() as session:
    session.run(init)

    print '==> restoring weights from '+weight_dir+'/task' + str(model.config.pach_id) + '.weights'
    #saver.restore(session, 'weights/task' + str(model.config.pach_id) + '.weights')
    saver.restore(session, weight_dir+'/task' + str(model.config.pach_id) + '.weights')
    #pdb.set_trace()
    print '==> running DMN'
    test_loss, test_accuracy, pred, ans, p = model.run_epoch(session, model.test)
    print ''
    #pdb.set_trace()
    print "answers:"
    print str(ans)
    print "pred:"
    print str(pred)
    valarr = (pred == ans)
    cnt = correct = 0
    Yes = model.ansList[0]
    for ii, a in enumerate(ans):
        if a==Yes and model.fnlist[ii].find("gen") == -1 : 
            cnt += 1 # count number of positive non-generated test cases
            if hasKeywords(key_words, model.fnlist[ii]) == 0:
                print "# overwrite prediction of "+model.fnlist[ii]+" to NO, containing no keywords."
                continue #skip files without key words. force pred to No
            if a==Yes and pred[ii]==Yes: # correctly predicted 
                correct += 1
                print model.fnlist[ii]

    print ''
    print "batch_size="+str(model.config.batch_size)
    print "non-generated positive correct: "+str(correct)
    print "non-generated positive cases  : "+str(cnt)
    if cnt > 0:
        print "non-generated positive accuracy: "+str(float(correct)/float(cnt))
    print "length of answers: "+str(len(ans))+" pred: "+str(len(pred))
    print "positive correct: "+str(np.sum((ans==Yes) & (Yes==pred))) #有  
    print "positive cases  : "+str(np.sum((ans==Yes)))
    if np.sum((ans==Yes)):
        print "positive accuracy: "+str(np.sum((ans==Yes) & (Yes==pred))/float(np.sum((ans==Yes))))
    if not model.config.eval_mode:
        No  = model.ansList[1]
        for ii, a in enumerate(ans):
            if a==No and model.fnlist[ii].find("gen") == -1 : 
                cnt += 1 # count number of positive non-generated test cases
                if a==No  and pred[ii]==No : # correctly predicted 
                    correct += 1
        print "non-generated negative correct: "+str(correct)
        print "non-generated negative cases  : "+str(cnt)
        if cnt > 0:
            print "non-generated negative accuracy: "+str(float(correct)/float(cnt))
        print "negtive  correct: "+str(np.sum((ans==No) & (No==pred))) #没有 
        print "negtive  cases  : "+str(np.sum((ans==No)))
        if np.sum((ans==No)):
            print "negtive  accuracy: "+str(np.sum((ans==No) & (No==pred))/float(np.sum((ans==No))))
    print "number of test cases="+str(len(ans))
    print "number of correct pred="+str(np.sum(pred==ans))
    accuracy = np.sum(pred == ans)/float(len(ans))
    print "my accuracy=", accuracy
    print 'Test accuracy:', test_accuracy
