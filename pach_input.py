# -*- coding: utf-8 -*-
import sys
import pdb
import os as os
import numpy as np
import logging

# can be sentence or word
input_mask_mode = "sentence"
datapath="/home/robert_tien/work/pachira/dmn/Pach_DMN/data/"
def init_pach(dir, question, answer, config):
    print "==> Loading test from %s" % dir
    tasks = []

    fnlist = []
    maxlen = 0
    maxfn=""
    #pdb.set_trace()
    for filename in os.listdir(dir):
    #logging.info(filename)
        #pdb.set_trace()
        fname = dir+"/"+filename
        f = open(fname,"r")
        fn = f.readline() # skip first line - meta info
        s = f.read()
        if not fn.startswith("FILE"):
            s = fn + s
        f.close()
        lines = repacklines(s, config, fname)
        if len(lines) > config.max_allowed_inputs:
            logging.info(str(len(lines))+" "+fname)
            print str(len(lines))+" "+fname
            continue
        if len(s.split(" ")) > maxlen:
           maxlen = len(s.split(" "))
           maxfn = filename
        if len(s) < 10: #24: 
            logging.info( "skip "+filename+": too short\n")
            print "skip "+filename+": too short\n"
            continue
        else:
            task = init_task(question, answer)
            #task = {"C": "", "Q": "", "A": "", "S": ""} 
            cnt = 0
            for l in lines:
                task["C"] += l
            task["Q"] = question
            task["A"] = answer
            task["S"] = []
            tasks.append(task.copy())
            fnlist.append(fname) # remember its filename
    if len(maxfn) > 0:
        logging.info(maxfn+" "+str(maxlen))
    print "number of tasks: "+str(len(tasks))
    return tasks, fnlist

# pack lines to equal number of words regardless of sentence structure since the sentence structure from input is noisy anyway.
# by doing this, we can reduce the number of lines and therefore the number of stages of rnn to avoid underfitting with too many variables.
def repacklines(s, config, fn):
    line = ""
    lines=[]
    cnt = 0
    for w in s.split():
        if w == ".":
            continue
        if cnt > 0 and cnt % config.max_allowed_sent_len == 0:
            lines.append(line+" . ")
            line = ""
        cnt += 1
        line += " "+w
    if len(line) > 0:
        lines.append(line+" . ")
    if config.repack and fn.find("Eval") > 0:
        f = open(fn+".rep","w")
        for line in lines:
            f.write(line+"\n")
        f.close()
    return lines

def split_line(line, config):
    lines = []
    words = line.split()
    aline = ""
    i = 0
    for w in words:
        i += 1
        if i % config.max_allowed_sent_len+1 == 0:
            lines.append(aline)
            aline=""
            i = 0
        aline += " "+w
    if len(aline) > 0:
        lines.append(aline)
    return lines
        
def init_task(question, answer):
    task = {"C": "", "Q": "", "A": "", "S": ""} 
    task["Q"] = question
    task["A"] = answer
    task["S"] = []
    return task

# adapted from https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano/
"""
def init_babi(fname):
    
    print "==> Loading test from %s" % fname
    tasks = []
    task = None
    for i, line in enumerate(open(fname)):
        id = int(line[0:line.find(' ')])
        if id == 1:
            task = {"C": "", "Q": "", "A": "", "S": ""} 
            counter = 0
            id_map = {}
            
        line = line.strip()
        line = line.replace('.', ' . ')
        line = line[line.find(' ')+1:]
        # if not a question
        if line.find('?') == -1:
            task["C"] += line
            id_map[id] = counter
            counter += 1
            
        else:
            idx = line.find('?')
            tmp = line[idx+1:].split('\t')
            task["Q"] = line[:idx]
            task["A"] = tmp[1].strip()
            task["S"] = []
            for num in tmp[2].split():
                task["S"].append(id_map[int(num.strip())])
            tasks.append(task.copy())

    return tasks
"""
def get_pach_raw_eval(config):
    pach_map = {
        "p1": {"D1":"dmnYesCarEval","Q":"客户 有 车 吗？","A1":"有","A2":"没有"},
        "p2": {"D1":"dmnYesHasInsEval","Q":"客户 有 保险 吗？","A1":"有","A2":"没有"},
        "p3": {"D1":"dmnYesHesitateEval","Q":"客户 是 怀疑 型 的 吗？","A1":"是","A2":"不是"}
    }
    ansList = []
    pach_dict = pach_map[config.pach_id]
    if config.eval_dir != None:
        pach_name = config.eval_dir
    else: 
        pach_name = pach_dict["D1"]
    question = pach_dict["Q"]
    answer = pach_dict["A1"]
    ansList.append(answer)
    pach_test_raw, fnlist = init_pach(os.path.join(datapath, pach_name),question, answer, config)
    answer = pach_dict["A2"]
    ansList.append(answer)
    print "all test categories tasks="+str(len(pach_test_raw))
    return pach_test_raw, fnlist, ansList
 
def get_pach_raw_test(config):
    pach_map = {
        "p1": {"D1":"dmnYesCarTest","D2":"dmnNoCarTest","D3":"all_unclass","Q":"客户 有 车 吗？","A1":"有","A2":"没有","A3":"有"},
        "p2": {"D1":"dmnYesHasInsTest","D2":"dmnNoHasInsTest","D3":"all_unclass","Q":"客户 有 保险 吗？","A1":"有","A2":"没有","A3":"有"},
        "p3": {"D1":"dmnYesHesitateTest","D2":"dmnNoHesitateTest","D3":"all_unclass","Q":"客户 是 怀疑 型 的 吗？","A1":"是","A2":"不是","A3":"是"}
    }
    ansList = []
    pach_dict = pach_map[config.pach_id]
    pach_name = pach_dict["D1"]
    question = pach_dict["Q"]
    answer = pach_dict["A1"]
    ansList.append(answer)
    pach_test_raw, fnlist = init_pach(os.path.join(datapath, pach_name),question, answer, config)
    pach_name = pach_dict["D2"]
    answer = pach_dict["A2"]
    ansList.append(answer)
    pach_test_raw2, fnlist2 = init_pach(os.path.join(datapath, pach_name),question, answer, config)
    pach_test_raw = pach_test_raw + pach_test_raw2
    fnlist = fnlist + fnlist2
    """
    pach_name = pach_dict["D3"]
    question = pach_dict["Q"]
    answer = pach_dict["A3"]
    pach_test_raw, fnlist = init_pach(os.path.join(datapath, pach_name),question, answer, config)
    """
    print "all test categories tasks="+str(len(pach_test_raw))
    return pach_test_raw, fnlist, ansList
    

def get_pach_raw(config):
    pach_map = {
            "p1": {"D1":"dmnYesCarTrain","D2":"dmnNoCarTrain","Q":"客户 有 车 吗？","A1":"有", "A2":"没有"},
            "p2": {"D1":"dmnYesHasInsTrain","D2":"dmnNoHasInsTrain","Q":"客户 有 保险 吗？","A1":"有", "A2":"没有"},
            "p3": {"D1":"dmnYesHesitateTrain","D2":"dmnNoHesitateTrain","D3":"all_unclass","Q":"客户 是 怀疑 型 的 吗？","A1":"是","A2":"不是","A3":"是"}
    }
    ansList = []
    pach_dict = pach_map[config.pach_id]
    pach_name = pach_dict["D1"]
    question = pach_dict["Q"]
    answer = pach_dict["A1"]
    ansList.append(answer)
    pach_train_raw, fnlist = init_pach(os.path.join(datapath, pach_name),question, answer, config)
    print "pach_train_raw length: "+str(len(pach_train_raw))
    print "fnlist length: "+str(len(fnlist))
    yes_pach_train_raw=[]
    yes_fnlist=[]
    for i in range(config.yes_repeat):
        yes_pach_train_raw += pach_train_raw
        yes_fnlist += fnlist
    print "yes_pach_train_raw length: "+str(len(yes_pach_train_raw))
    print "yes_fnlist length: "+str(len(yes_fnlist))
    pach_name = pach_dict["D2"]
    answer = pach_dict["A2"]
    ansList.append(answer)
    pach_train_raw2, fnlist2 = init_pach(os.path.join(datapath, pach_name),question, answer, config)
    pach_train_raw = yes_pach_train_raw + pach_train_raw2
    fnlist = yes_fnlist + fnlist2
    print "all train categories tasks="+str(len(pach_train_raw))
    #pach_test_raw, fnlist = init_pach(os.path.join(datapath, pach_name+"_test"))
    return pach_train_raw, fnlist, ansList

            
def load_glove(config):
    word2vec = {}
    vocab={}
    ivocab={}
    
    print "==> loading glove"
    #pdb.set_trace()
    #with open(("./data/glove/glove.6B/glove.6B." + str(config.embed_size) + "d.txt")) as f:
    with open(("/home/robert_tien/work/GloVe/wiki_allcat_news_web_"+str(config.vec_size)+"/vectors.txt")) as f:
    #with open(("/home/robert_tien/work/GloVe/wiki_allcat_news_web_100/vectors.txt")) as f:
        for line in f:
            l = line.split()
            if l[0][0].isalpha():
                continue
            word2vec[l[0]] = map(float, l[1:])
    with open(("/home/robert_tien/work/GloVe/wiki_allcat_news_web_"+str(config.vec_size)+"/vocab.txt")) as f:
        vocab["<unk>"]=0
        ivocab[0]="<unk>"
        create_vector("<unk>", word2vec, config.embed_size)
        i = 1
        for line in f:
            l = line.split()
            cnt = int(l[1])
            if l[0][0].isalpha():
                continue
            if cnt < config.min_vocab_cnt:
                del word2vec[l[0]]
                continue
            vocab[l[0]] = i
            ivocab[i]=l[0]
            i += 1
            
    print "==> glove is loaded: vocab length="+str(len(vocab))+" i="+str(i)
    
    return word2vec, vocab, ivocab


def create_vector(word, word2vec, word_vector_size, silent=True):
    # if the word is missing from Glove, create some fake vector and store in glove!
    vector = np.random.uniform(0.0,1.0,(word_vector_size,))
    word2vec[word] = vector
    if (not silent):
        print "utils.py::create_vector => %s is missing" % word
    return vector

def process_word(config, word, word2vec, vocab, ivocab, word_vector_size, to_return="word2vec", silent=True):
    if config.word2vec_init:
        if not word in word2vec:
            word = "<unk>"
    else: 
        if not word in word2vec:
            create_vector(word, word2vec, word_vector_size, silent)
        if not word in vocab: 
            next_index = len(vocab)
            vocab[word] = next_index
            ivocab[next_index] = word
    
    if to_return == "word2vec":
        return word2vec[word]
    elif to_return == "index":
        return vocab[word]
    elif to_return == "onehot":
        raise Exception("to_return = 'onehot' is not implemented yet")

def process_input(config, data_raw, floatX, word2vec, vocab, ivocab, embed_size, split_sentences=False):
    questions = []
    inputs = []
    answers = []
    input_masks = []
    relevant_labels = []
    for x in data_raw:
        if split_sentences:
            inp = x["C"].lower().split(' . ') 
            inp = [w for w in inp if len(w) > 0]
            inp = [i.split() for i in inp]
        else:
            inp = x["C"].lower().split(' ') 
            inp = [w for w in inp if len(w) > 0]

        q = x["Q"].lower().split(' ')
        q = [w for w in q if len(w) > 0]

        if split_sentences: 
            inp_vector = [[process_word(config, word = w, 
                                        word2vec = word2vec, 
                                        vocab = vocab, 
                                        ivocab = ivocab, 
                                        word_vector_size = embed_size, 
                                        to_return = "index") for w in s] for s in inp]
        else:
            inp_vector = [process_word(config, word = w, 
                                        word2vec = word2vec, 
                                        vocab = vocab, 
                                        ivocab = ivocab, 
                                        word_vector_size = embed_size, 
                                        to_return = "index") for w in inp]
                                    
        q_vector = [process_word(config, word = w, 
                                    word2vec = word2vec, 
                                    vocab = vocab, 
                                    ivocab = ivocab, 
                                    word_vector_size = embed_size, 
                                    to_return = "index") for w in q]
        
        if split_sentences:
            inputs.append(inp_vector)
        else:
            inputs.append(np.vstack(inp_vector).astype(floatX))
        questions.append(np.vstack(q_vector).astype(floatX))
        answers.append(process_word(config, word = x["A"], 
                                        word2vec = word2vec, 
                                        vocab = vocab, 
                                        ivocab = ivocab, 
                                        word_vector_size = embed_size, 
                                        to_return = "index"))
        # NOTE: here we assume the answer is one word! 

        if not split_sentences:
            if input_mask_mode == 'word':
                input_masks.append(np.array([index for index, w in enumerate(inp)], dtype=np.int32)) 
            elif input_mask_mode == 'sentence': 
                input_masks.append(np.array([index for index, w in enumerate(inp) if w == '.'], dtype=np.int32)) 
            else:
                raise Exception("invalid input_mask_mode")

        relevant_labels.append(x["S"])
    
    return inputs, questions, answers, input_masks, relevant_labels 

def get_lens(inputs, split_sentences=False):
    lens = np.zeros((len(inputs)), dtype=int)
    for i, t in enumerate(inputs):
        lens[i] = t.shape[0]
    return lens

def get_sentence_lens(inputs):
    lens = np.zeros((len(inputs)), dtype=int)
    sen_lens = []
    max_sen_lens = []
    for i, t in enumerate(inputs):
        sentence_lens = np.zeros((len(t)), dtype=int)
        for j, s in enumerate(t):
            sentence_lens[j] = len(s)
        lens[i] = len(t)
        sen_lens.append(sentence_lens)
        max_sen_lens.append(np.max(sentence_lens))
    return lens, sen_lens, max(max_sen_lens)
    

def pad_inputs(inputs, lens, max_len, mode="", sen_lens=None, max_sen_len=None):
    if mode == "mask":
        padded = [np.pad(inp, (0, max_len - lens[i]), 'constant', constant_values=0) for i, inp in enumerate(inputs)]
        return np.vstack(padded)

    elif mode == "split_sentences":
        padded = np.zeros((len(inputs), max_len, max_sen_len))
        for i, inp in enumerate(inputs):
            padded_sentences = [np.pad(s, (0, max_sen_len - sen_lens[i][j]), 'constant', constant_values=0) for j, s in enumerate(inp)]
            # trim array according to max allowed inputs
            if len(padded_sentences) > max_len: # this should not happen!!
                #pdb.set_trace()
                assert(padded_sentences == max_len)
                padded_sentences = padded_sentences[(len(padded_sentences)-max_len):]
                lens[i] = max_len
            padded_sentences = np.vstack(padded_sentences)
            padded_sentences = np.pad(padded_sentences, ((0, max_len - lens[i]),(0,0)), 'constant', constant_values=0)
            padded[i] = padded_sentences
        return padded

    padded = [np.pad(np.squeeze(inp, axis=1), (0, max_len - lens[i]), 'constant', constant_values=0) for i, inp in enumerate(inputs)]
    return np.vstack(padded)

def create_embedding(word2vec, ivocab, embed_size):
    embedding = np.zeros((len(ivocab), embed_size))
    for i in range(len(ivocab)):
        word = ivocab[i]
        embedding[i] = word2vec[word]
    return embedding

def dump_raw_data(tasks, fname):
    f=open(fname,"w")
    for i, task in enumerate(tasks):
        f.write("content: "+str(i)+"\n")
        for j,l in enumerate(task["C"].split('.')):
            f.write(str(j)+": ")
            f.write(l)
            f.write("\n")
        f.write("Question:\n")
        f.write(task["Q"])
        f.write("\n")
        f.write("Answer:\n")
        f.write(task["A"])
        f.write("\n")
    f.close()
    return
        
def load_pach(config, split_sentences=False):
    vocab = {}
    ivocab = {}

    #pdb.set_trace()
    pach_train_raw, fnlist0, ansList0 = get_pach_raw(config)
    pach_test_raw, fnlist1, ansList1 = get_pach_raw_test(config) if not config.eval_mode else get_pach_raw_eval(config)
    if config.train_mode:
        fnlist = fnlist0
        ansList = ansList0
    else:
        fnlist = fnlist1
        ansList = ansList1
        dump_raw_data(pach_test_raw, "pach_test_raw.dmp")
        print "finish dumping raw test/eval data"
        """
        import json
        encoded = json.dumps(pach_test_raw)
        f = open("pach_test_raw.dmp","w")
        f.write(encoded)
        f.close()
        """

    if config.word2vec_init:
        #assert config.embed_size == 100
        word2vec, vocab, ivocab = load_glove(config)
    else:
        word2vec = {}

    # set word at index zero to be end of sentence token so padding with zeros is consistent
    process_word(config, word = "<eos>", 
                word2vec = word2vec, 
                vocab = vocab, 
                ivocab = ivocab, 
                word_vector_size = config.embed_size, 
                to_return = "index")

    print '==> get train inputs'
    train_data = process_input(config, pach_train_raw, config.floatX, word2vec, vocab, ivocab, config.embed_size, split_sentences)
    print '==> get test inputs'
    test_data = process_input(config, pach_test_raw, config.floatX, word2vec, vocab, ivocab, config.embed_size, split_sentences)

    if config.word2vec_init:
        #assert config.embed_size == 100
        word_embedding = create_embedding(word2vec, ivocab, config.embed_size)
    else:
        #word_embedding = np.random.uniform(-config.embedding_init, config.embedding_init, (config.embedding_len, config.embed_size))
        word_embedding = np.random.uniform(-config.embedding_init, config.embedding_init, (len(ivocab), config.embed_size))
    print "vocab len="+str(len(vocab))

    inputs, questions, answers, input_masks, rel_labels = train_data if config.train_mode else test_data

    if split_sentences:
        input_lens, sen_lens, max_sen_len = get_sentence_lens(inputs)
        max_mask_len = max_sen_len
    else:
        input_lens = get_lens(inputs)
        mask_lens = get_lens(input_masks)
        max_mask_len = np.max(mask_lens)

    q_lens = get_lens(questions)

    max_q_len = np.max(q_lens)
    max_input_len = min(np.max(input_lens), config.max_allowed_inputs)

    #pad out arrays to max
    if split_sentences:
        inputs = pad_inputs(inputs, input_lens, max_input_len, "split_sentences", sen_lens, max_sen_len)
        input_masks = np.zeros(len(inputs))
    else:
        inputs = pad_inputs(inputs, input_lens, max_input_len)
        input_masks = pad_inputs(input_masks, mask_lens, max_mask_len, "mask")

    questions = pad_inputs(questions, q_lens, max_q_len)

    answers = np.stack(answers)

    rel_labels = np.zeros((len(rel_labels), len(rel_labels[0])))

    for i, tt in enumerate(rel_labels):
        rel_labels[i] = np.array(tt, dtype=int)

    #pdb.set_trace()
    if config.train_mode:
        nrInp = len(questions)
        p = np.random.permutation(nrInp)
        questions, inputs, q_lens, input_lens, input_masks, answers, rel_labels = questions[p], inputs[p], q_lens[p], input_lens[p], input_masks[p], answers[p], rel_labels[p] 
        fnlist = [fnlist[i] for i in p]
        nrTrain = nrInp - nrInp/10
        print "training: "+str(nrTrain)+" validation: "+str(nrInp - nrTrain)
        train = questions[:nrTrain], inputs[:nrTrain], q_lens[:nrTrain], input_lens[:nrTrain], input_masks[:nrTrain], answers[:nrTrain], rel_labels[:nrTrain] 

        valid = questions[nrTrain:], inputs[nrTrain:], q_lens[nrTrain:], input_lens[nrTrain:], input_masks[nrTrain:], answers[nrTrain:], rel_labels[nrTrain:] 
        return train, valid, word_embedding, max_q_len, max_input_len, max_mask_len, rel_labels.shape[1], len(vocab), fnlist,[vocab[ansList[0]],vocab[ansList[1]]]

    else:
        test = questions, inputs, q_lens, input_lens, input_masks, answers, rel_labels
        return test, word_embedding, max_q_len, max_input_len, max_mask_len, rel_labels.shape[1], len(vocab), fnlist, [vocab[ansList[0]],vocab[ansList[1]]], pach_test_raw


    
