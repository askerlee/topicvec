# -*- coding=GBK -*-

from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import reuters
import HTMLParser
import os
import sys
import unicodedata
import re
import pdb

unicode_punc_tbl = dict.fromkeys( i for i in xrange(128, sys.maxunicode)
                      if unicodedata.category(unichr(i)).startswith('P') )

# 输入: 一个文档
# 处理过程: 先按标点符号分成句子，然后每句按词边界分词
def extractSentenceWords(doc, remove_url=True, remove_punc="utf-8", min_length=1):
    # 去掉指定字符集(缺省去掉utf-8的)中的标点符号
    if remove_punc:
        # ensure doc_u is in unicode
        if not isinstance(doc, unicode):
            encoding = remove_punc
            doc_u = doc.decode(encoding)
        else:
            doc_u = doc
        # remove unicode punctuation marks, keep ascii punctuation marks
        doc_u = doc_u.translate(unicode_punc_tbl)
        if not isinstance(doc, unicode):
            doc = doc_u.encode(encoding)
        else:
            doc = doc_u
    
    # 去掉文本中的URL(可选)        
    if remove_url:
        re_url = r"(https?:\/\/)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)"
        doc = re.sub( re_url, "", doc )
    
    # 按句子标点分句        
    sentences = re.split( r"\s*[,;:`\"()?!{}]\s*|--+|\s*-\s+|''|\.\s|\.$|\.\.+|“|”", doc ) #"
    wc = 0
    wordsInSentences = []
    
    for sentence in sentences:
        if sentence == "":
            continue

        if not re.search( "[A-Za-z0-9]", sentence ):
            continue

        # 按词边界分词
        words = re.split( r"\s+\+|^\+|\+?[\-*\/&%=<>\[\]~\|\@\$]+\+?|\'\s+|\'s\s+|\'s$|\s+\'|^\'|\'$|\$|\\|\s+", sentence )

        words = filter( lambda w: w, words )

        if len(words) >= min_length:
            wordsInSentences.append(words)
            wc += len(words)

    #print "%d words extracted" %wc
    return wordsInSentences, wc
    
def load_20news(setName):
    newsgroups_subset = fetch_20newsgroups(subset=setName, remove=('headers', 'footers')) #, 'quotes'
    totalLineNum = 0
    readDocNum = 0
    print "Loading 20 newsgroup %s data..." %setName
        
    setDocNum = len(newsgroups_subset.data)
    orig_docs_name = []
    orig_docs_cat = []
    orig_docs_words = []
    
    catNum = len(newsgroups_subset.target_names)
    cats_docsWords = [ [] for i in xrange(catNum) ]
    cats_docNames = [ [] for i in xrange(catNum) ]
    
    emptyFileNum = 0
    
    for d, text in enumerate(newsgroups_subset.data):
        if d % 50 == 49 or d == setDocNum - 1:
            print "\r%d %d\r" %( d + 1, totalLineNum ),
        text = text.encode("utf-8")
        lines = text.split("\n")
        if len(text) == 0 or len(lines) == 0:
            emptyFileNum += 1
            continue
    
        readDocNum += 1
        totalLineNum += len(lines)
    
        catID = newsgroups_subset.target[d]
        category = newsgroups_subset.target_names[catID]
    
        text = " ".join(lines)
    
        wordsInSentences, wc = extractSentenceWords(text)
        filename = newsgroups_subset.filenames[d]
        filename = os.path.basename(filename)
        orig_docs_words.append( wordsInSentences )
        orig_docs_name.append(filename)
        orig_docs_cat.append(catID)
        cats_docsWords[catID].append(wordsInSentences)
        cats_docNames[catID].append(filename)
    
    print "Done. %d docs read, %d empty docs skipped. Totally %d lines" %(readDocNum, emptyFileNum, totalLineNum)
    return setDocNum, orig_docs_words, orig_docs_name, orig_docs_cat, \
                cats_docsWords, cats_docNames, newsgroups_subset.target_names
    
def load_reuters(setName):
    html = HTMLParser.HTMLParser()
    doc_ids = reuters.fileids()
    cat2all_ids = {}
    cat2train_ids = {}
    cat2test_ids = {}
    cat2all_num = {}
    cand_docNum = 0
    
    for doc_id in doc_ids:
        # only choose docs belonging in one category
        if len( reuters.categories(doc_id) ) == 1:
            cat = reuters.categories(doc_id)[0]
            cand_docNum += 1
            
            # 以'train'开头的文档名放在training集合里
            if doc_id.startswith("train"):
                cat2set_ids = cat2train_ids
            # 否则，放到test集合
            else:
                cat2set_ids = cat2test_ids
                
            if cat in cat2set_ids:
                cat2set_ids[cat].append(doc_id)
            else:
                cat2set_ids[cat] = [ doc_id ]
            
            # both train and test doc_ids are put in cat2all_ids
            if cat in cat2all_ids:
                cat2all_ids[cat].append(doc_id)
            else:
                cat2all_ids[cat] = [ doc_id ]
            if cat in cat2all_num:
                cat2all_num[cat] += 1
            else:
                cat2all_num[cat] = 1
            
    print "Totally %d docs, %d single-category docs in %d categories" %( len(doc_ids), 
                    cand_docNum, len(cat2train_ids) )
                    
    sorted_cats = sorted( cat2all_num.keys(), key=lambda cat: cat2all_num[cat],
                            reverse=True )
                            
    catNum = 10
    cats_docsWords = [ [] for i in xrange(catNum) ]
    cats_docNames = [ [] for i in xrange(catNum) ]
                            
    topN_cats = sorted_cats[:catNum]
    print "Top 10 categories:"
    keptAllDocNum = 0
    keptTrainDocNum = 0
    keptTestDocNum = 0
    
    for cat in topN_cats:
        print "%s: %d/%d" %( cat, len(cat2train_ids[cat]), len(cat2test_ids[cat]) )
        keptTrainDocNum += len(cat2train_ids[cat])
        keptTestDocNum += len(cat2test_ids[cat])
        keptAllDocNum += len(cat2train_ids[cat]) + len(cat2test_ids[cat])
        
    print "Totally %d docs kept, %d in train, %d in test" %( keptAllDocNum, 
                        keptTrainDocNum, keptTestDocNum )    
    
    if setName == "train":
        cat2set_ids = cat2train_ids
        setDocNum = keptTrainDocNum
    elif setName == "test":
        cat2set_ids = cat2test_ids
        setDocNum = keptTestDocNum
    elif setName == "all":
        cat2set_ids = cat2all_ids
        setDocNum = keptAllDocNum
    else:
        raise Exception("Unknown set name %s" %setName)
            
    orig_docs_name = []
    orig_docs_cat = []
    orig_docs_words = []
    readDocNum = 0
    totalLineNum = 0
    emptyFileNum = 0
    
    for cat_id, cat in enumerate(topN_cats):
        for doc_id in cat2set_ids[cat]:
            if readDocNum % 50 == 49 or readDocNum == setDocNum - 1:
                print "\r%d %d\r" %( readDocNum + 1, totalLineNum ),
            text = html.unescape( reuters.raw(doc_id) )
            text = text.encode("utf-8")
            lines = text.split("\n")
            if len(text) == 0 or len(lines) == 0:
                emptyFileNum += 1
                continue
        
            readDocNum += 1
            totalLineNum += len(lines)
        
            text = " ".join(lines)
            wordsInSentences, wc = extractSentenceWords(text)
            
            filename = doc_id
            orig_docs_words.append( wordsInSentences )
            orig_docs_name.append(filename)
            orig_docs_cat.append(cat_id)
            cats_docsWords[cat_id].append(wordsInSentences)
            cats_docNames[cat_id].append(filename)
            
    print "Done. %d docs read, %d empty docs skipped. Totally %d lines" %(readDocNum, emptyFileNum, totalLineNum)
    return setDocNum, orig_docs_words, orig_docs_name, orig_docs_cat, \
                cats_docsWords, cats_docNames, topN_cats
    