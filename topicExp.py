import sys
import pdb
import os
import getopt
from corpusLoader import *
from utils import *
from topicvecDir import topicvecDir

config = dict(  unigramFilename = "top1grams-wiki.txt",
                word_vec_file = "25000-180000-500-BLK-8.0.vec",
                #word_vec_file = "word2vec.vec",
                load_embedding_word_count = 180000,
                K = 100,
                # for separate category training, each category has 10 topics, totalling 200
                sepK_20news = 15,
                sepK_reuters = 12,
                # set it to 0 to disable the removal of very small topics
                topTopicMassFracThres = 0.05,
                N0 = 500,
                # don't set it too big, e.g. 5.
                # otherwise derived topics will be too specific, and classification accuracy will drop.
                max_l = 3,
                init_l = 1,
                # cap the norm of the gradient of topics to avoid too big gradients
                max_grad_norm = 1,
                Mstep_sample_topwords = 25000,
                # normalize by the sum of Em when updating topic embeddings
                # to avoid too big gradients
                grad_scale_Em_base = 20000,
                topW = 12,
                # when topTopicMassFracPrintThres = 0, print all topics
                topTopicMassFracPrintThres = 0,
                alpha0 = 0.1,
                alpha1 = 0.1,
                delta = 0.1,
                MAX_EM_ITERS = 200,
                MAX_TopicProp_ITERS = 1,
                topicDiff_tolerance = 1e-2,
                zero_topic0 = True,
                remove_stop = True,
                useDrdtApprox = False,
                verbose = 0,
                seed = 0,
                printTopics_iterNum = 10,
                calcSum_pi_v_iterNum = 1,
                VStep_iterNum = 5
            )

def usage():
    print """Usage: topicExp.py -s                corpus_name set_name(s)
                   -i topic_vec_file corpus_name set_name(s)
                   [ -w ]            corpus_name set_name(s)
                   (Optional) -t max_iter_num ...
  corpus_name: '20news' or  'reuters'
  set_name(s): 'train', 'test' or 'train,test' (will save in separate files)
  -s:          Train on separate categories
  -i:          Do inference on a corpus given a topic vec file
  -w:          Dump words only (no inference of topics)
  -t:          Specify the maximum number of iterations"""
  
corpusName = None
corpus2loader = { '20news': load_20news, 'reuters': load_reuters }
    
subsetNames = [ ]
topic_vec_file = None
MAX_ITERS = -1
onlyDumpWords = False
separateCatTraining = False
onlyInferTopicProp = False
topicTraitStr = ""
onlyGetOriginalText = False

try:
    opts, args = getopt.getopt( sys.argv[1:], "i:t:wso" )

    if len(args) == 0:
        raise getopt.GetoptError("Not enough free arguments")
    corpusName = args[0]
    if len(args) == 2:
        subsetNames = args[1].split(",")
    if len(args) > 2:
        raise getopt.GetoptError("Too many free arguments")

    for opt, arg in opts:
        if opt == '-i':
            onlyInferTopicProp = True
            topic_vec_file = arg
            # if 'useDrdtApprox' == True, will precompute matrix Evv, which is very slow
            # disable to speed up
            config['useDrdtApprox'] = False
        if opt == '-t':
            MAX_ITERS = int(arg)
        if opt == '-w':
            onlyDumpWords = True
            # if 'useDrdtApprox' == True, will precompute matrix Evv, which is very slow
            # disable to speed up
            config['useDrdtApprox'] = False
        if opt == '-s':
            separateCatTraining = True
        if opt == '-o':
            onlyGetOriginalText = True
            
except getopt.GetoptError, e:
    print e.msg
    usage()
    sys.exit(2)

if not onlyGetOriginalText:
# The leading 'all-mapping' is only to get word mappings from the original IDs in 
# the embedding file to a compact word ID list, to speed up computation of sLDA
# The mapping has to be done on 'all' to include all words in train and test sets
    subsetNames = [ 'all-mapping' ] + subsetNames
    
if MAX_ITERS > 0:
    if onlyInferTopicProp:
        MAX_TopicProp_ITERS = MAX_ITERS
    else:
        config['MAX_EM_ITERS'] = MAX_ITERS

loader = corpus2loader[corpusName]
wid2compactId = {}
compactId_words = []
hasIdMapping = False

if onlyInferTopicProp:
    topicfile_trunk = topic_vec_file.split(".")[0]
    topicTraits = topicfile_trunk.split("-")[3:]
    topicTraitStr = "-".join(topicTraits)
    T = load_matrix_from_text( topic_vec_file, "topic" )
    config['K'] = T.shape[0]

config['logfilename'] = corpusName
topicvec = topicvecDir(**config)
out = topicvec.genOutputter(0)

for si, subsetName in enumerate(subsetNames):       
    print "Process subset '%s':" %subsetName
    if subsetName == 'all-mapping':
        subsetName = 'all'
        onlyGetWidMapping = True
    else:
        onlyGetWidMapping = False
        
    subsetDocNum, orig_docs_words, orig_docs_name, orig_docs_cat, cats_docsWords, \
            cats_docNames, category_names = loader(subsetName)
    catNum = len(category_names)
    basename = "%s-%s-%d" %( corpusName, subsetName, subsetDocNum )

    # dump original words (without filtering)
    orig_filename = "%s.orig.txt" %basename
    ORIG = open( orig_filename, "w" )
    for wordsInSentences in orig_docs_words:
        for sentence in wordsInSentences:
            for w in sentence:
                w = w.lower()        
                ORIG.write( "%s " %w )
        ORIG.write("\n")
    ORIG.close()
    print "%d original docs saved in '%s'" %( subsetDocNum, orig_filename )

    if onlyGetOriginalText:
        continue
        
    docs_idx = topicvec.setDocs( orig_docs_words, orig_docs_name )
    docs_name = [ orig_docs_name[i] for i in docs_idx ]
    docs_cat = [ orig_docs_cat[i] for i in docs_idx ]
    readDocNum = len(docs_idx)
    out( "%d docs left after filtering empty docs" %(readDocNum) )
    assert readDocNum == topicvec.D, "Returned %d doc idx != %d docs in Topicvec" %(readDocNum, topicvec.D)
    
    # executed when subsetName == 'all-mapping'
    if onlyGetWidMapping:
        sorted_wids = sorted( topicvec.wid2freq.keys() )
        uniq_wid_num = len(sorted_wids)
        for i, wid in enumerate(sorted_wids):
            # svm feature index cannot be 0
            # +1 to avoid 0 being used as a feature index
            wid2compactId[wid] = i + 1
            compactId_words.append( topicvec.vocab[wid] )
            
        hasIdMapping = True
        onlyGetWidMapping = False
        print "Word mapping created: %d -> %d" %( sorted_wids[-1], uniq_wid_num )
        id2word_filename = "%s.id2word.txt" %basename
        ID2WORD = open( id2word_filename, "w" )
        for i in xrange(uniq_wid_num):
            ID2WORD.write( "%d\t%s\n" %( i, compactId_words[i] ) )
        ID2WORD.close()
        continue
            
    # dump words in stanford classifier format
    stanford_filename = "%s.stanford-bow.txt" %basename
    STANFORD = open( stanford_filename, "w" )
    for i in xrange(readDocNum):
        wids = topicvec.docs_wids[i]
        words = [ topicvec.vocab[j] for j in wids ]
        text = " ".join(words)
        catID = docs_cat[i]
        category = category_names[catID]
        doc_name = docs_name[i]
        STANFORD.write( "%s\t%s\t%s\n" %( category, doc_name, text ) )
    
    STANFORD.close()
    print "%d docs saved in '%s' in stanford bow format" %( readDocNum, stanford_filename )

    # dump words in sLDA format
    slda_bow_filename = "%s.slda-bow.txt" %basename
    slda_label_filename = "%s.slda-label.txt" %basename
    SLDA_BOW = open( slda_bow_filename, "w" )
    SLDA_LABEL = open( slda_label_filename, "w" )
    
    for i in xrange(readDocNum):
        wids = topicvec.docs_wids[i]
        # compact wid to freq
        cwid2freq = {}
        for wid in wids:
            cwid = wid2compactId[wid]
            if cwid in cwid2freq:
                cwid2freq[cwid] += 1
            else:
                cwid2freq[cwid] = 1
        catID = docs_cat[i]
        sorted_cwids = sorted( cwid2freq.keys() )
        uniq_wid_num = len(sorted_cwids)
        # sLDA requires class lables to start from 0
        SLDA_LABEL.write( "%d\n" %catID )
        SLDA_BOW.write( "%d" %uniq_wid_num )
        for cwid in sorted_cwids:
            SLDA_BOW.write( " %d:%d" %( cwid, cwid2freq[cwid] ) )
        SLDA_BOW.write("\n")
            
    SLDA_BOW.close()
    SLDA_LABEL.close()
    
    print "%d docs saved in '%s' and '%s' in sLDA bow format" %( readDocNum, 
                slda_bow_filename, slda_label_filename )
        
    # dump words in libsvm/svmlight format
    svmbow_filename = "%s.svm-bow.txt" %basename
    SVMBOW = open( svmbow_filename, "w" )
    for i in xrange(readDocNum):
        wids = topicvec.docs_wids[i]
        cwid2freq = {}
        for wid in wids:
            cwid = wid2compactId[wid]
            if cwid in cwid2freq:
                cwid2freq[cwid] += 1
            else:
                cwid2freq[cwid] = 1
        catID = docs_cat[i]
        sorted_cwids = sorted( cwid2freq.keys() )
        SVMBOW.write( "%d" %(catID+1) )
        for cwid in sorted_cwids:
            SVMBOW.write( " %d:%d" %( cwid, cwid2freq[cwid] ) )
        SVMBOW.write("\n")
    
    SVMBOW.close()
    print "%d docs saved in '%s' in svm bow format" %( readDocNum, svmbow_filename )
        
    if onlyDumpWords:
        continue
    
    # load topics from a file, infer the topic proportions, and save the proportions
    if onlyInferTopicProp:
        docs_Em, docs_Pi = topicvec.inferTopicProps(T, config['MAX_TopicProp_ITERS'])
        # dump the topic proportions in my own matrix format
        save_matrix_as_text( basename + "-%s-i%d.topic.prop" %(topicTraitStr, config['MAX_TopicProp_ITERS']), 
                                "topic proportion", docs_Em, docs_cat, docs_name, colSep="\t" )
        
        # dump the topic proportions into SVMTOPIC_PROP in libsvm/svmlight format
        # dump the mix of word freqs and topic proportions into SVMTOPIC_BOW in libsvm/svmlight format   
        svmtopicprop_filename = "%s.svm-topicprop.txt" %basename
        # topic props + weighted sum of topic vectors
        svmtopicbow_filename = "%s.svm-topicbow.txt" %basename
        svmtopic_wvavg_filename = "%s.svm-topic-wvavg.txt" %basename
        
        SVMTOPIC_PROP = open( svmtopicprop_filename, "w" )
        SVMTOPIC_BOW = open( svmtopicbow_filename, "w" )
        SVMTOPIC_WVAVG = open( svmtopic_wvavg_filename, "w" )
        
        wordvec_avg = np.zeros( topicvec.N0 )
        
        for i in xrange(readDocNum):
            wids = topicvec.docs_wids[i]
            cwid2freq = {}
            for wid in wids:
                cwid = wid2compactId[wid]
                if cwid in cwid2freq:
                    cwid2freq[cwid] += 1
                else:
                    cwid2freq[cwid] = 1
                
                wordvec_avg += topicvec.V[wid]
                    
            catID = docs_cat[i]
            sorted_cwids = sorted( cwid2freq.keys() )
            
            SVMTOPIC_PROP.write( "%d" %(catID+1) )
            SVMTOPIC_BOW.write( "%d" %(catID+1) )
            SVMTOPIC_WVAVG.write( "%d" %(catID+1) )
            
            for k in xrange(topicvec.K):
                SVMTOPIC_PROP.write( " %d:%.3f" %( k+1, docs_Em[i][k] ) )
                SVMTOPIC_BOW.write( " %d:%.3f" %( k+1, docs_Em[i][k] ) )
                SVMTOPIC_WVAVG.write( " %d:%.3f" %( k+1, docs_Em[i][k] ) )
            
            for cwid in sorted_cwids:
                # first K indices are reserved for topic features, so add topicvec.K here
                SVMTOPIC_BOW.write( " %d:%d" %( cwid + topicvec.K, cwid2freq[cwid] ) )
            
            wordvec_avg /= topicvec.docs_L[i]
            for n in xrange(topicvec.N0):
                SVMTOPIC_WVAVG.write( " %d:%.3f" %( n + 1 + topicvec.K, wordvec_avg[n] ) )
                
            SVMTOPIC_PROP.write("\n")    
            SVMTOPIC_BOW.write("\n")
            SVMTOPIC_WVAVG.write("\n")
            
        SVMTOPIC_PROP.close()
        SVMTOPIC_BOW.close()
        SVMTOPIC_WVAVG.close()
        
        print "%d docs saved in '%s' in svm topicProp format" %( readDocNum, svmtopicprop_filename )
        print "%d docs saved in '%s' in svm topicProp-BOW format" %( readDocNum, svmtopicbow_filename )
        print "%d docs saved in '%s' in svm topicProp-WordvecAvg format" %( readDocNum, svmtopic_wvavg_filename )
        
    # infer topics from docs, and save topics and their proportions in each doc
    else:
        if not separateCatTraining:
            best_last_Ts, Em, docs_Em, Pi = topicvec.inference()

            best_it, best_T, best_loglike = best_last_Ts[0]
            last_it, last_T, last_loglike = best_last_Ts[1]
            
            save_matrix_as_text( basename + "-em%d-best.topic.vec" %best_it, "best topics", best_T  )
            save_matrix_as_text( basename + "-em%d-last.topic.vec" %last_it, "last topics", last_T  )
                
            save_matrix_as_text( basename + "-em%d.topic.prop" %config['MAX_EM_ITERS'], "topic proportion", docs_Em, docs_cat, docs_name, colSep="\t" )

        else:
            # infer topics for each category, combine them and save in one file
            if corpusName == "20news":
                topicvec.setK( config['sepK_20news'] )
            else:
                topicvec.setK( config['sepK_reuters'] )
                
            best_T = []
            last_T = []
            slim_T = []
            totalDocNum = 0
            #pdb.set_trace()
            
            for catID in xrange(catNum):
                out("")
                out( "Inference on category %d:" %( catID+1 ) )
                cat_docs_idx = topicvec.setDocs( cats_docsWords[catID], cats_docNames[catID] )
                totalDocNum += len(cat_docs_idx)
                cat_best_last_Ts, cat_Em, cat_docs_Em, cat_Pi = topicvec.inference()
                cat_best_it, cat_best_T, cat_best_loglike = cat_best_last_Ts[0]
                if cat_best_last_Ts[1]:
                    cat_last_it, cat_last_T, cat_last_loglike = cat_best_last_Ts[1]
                else:
                    cat_last_it, cat_last_T, cat_last_loglike = cat_best_last_Ts[0]
                    
                # normalize by the number of documents 
                cat_Em2 = cat_Em / len(cat_docs_Em)
                
                if catID > 0 and config['zero_topic0']:
                    # remove the redundant null topic
                    removeNullTopic = True
                    best_T.append( cat_best_T[1:] )
                    last_T.append( cat_last_T[1:] )
                else:
                    # keep null topic
                    removeNullTopic = False
                    best_T.append( cat_best_T )
                    last_T.append( cat_last_T )
 
                sorted_tids = sorted( range(topicvec.K), key=lambda k: cat_Em[k], reverse=True )
                out("Topic normalized mass:")
                s = ""
                for tid in sorted_tids:
                    s += "%d: %.3f " %( tid, cat_Em2[tid] )
                out(s)
                
                if config['topTopicMassFracThres'] > 0:
                    cat_Em2_thres = np.sum(cat_Em2) / topicvec.K * config['topTopicMassFracThres']
                    out( "Topic normalized mass thres: %.3f" %cat_Em2_thres )
                    top_tids = []
                    for i,tid in enumerate(sorted_tids):
                        if cat_Em2[tid] <= cat_Em2_thres:
                            break
                        if removeNullTopic and tid == 0:
                            continue
                        top_tids.append(tid)
                        
                    out( "Keep top %d topics:" %len(top_tids) )
                    s = ""
                    for tid in top_tids:
                        s += "%d: %.3f " %( tid, cat_Em2[tid] )
                    out(s)
                    
                    slim_cat_T = cat_last_T[top_tids]
                    slim_T.append(slim_cat_T)
                
            out( "Done inference on %d docs in %d categories" %(totalDocNum, catNum) )

            best_T = np.concatenate(best_T)
            last_T = np.concatenate(last_T)
            save_matrix_as_text( "%s-sep%d-em%d-best.topic.vec" %( basename, best_T.shape[0], 
                                            topicvec.MAX_EM_ITERS ), "best topics", best_T )
            save_matrix_as_text( "%s-sep%d-em%d-last.topic.vec" %( basename, last_T.shape[0], 
                                            topicvec.MAX_EM_ITERS ), "last topics", last_T )

            if config['topTopicMassFracThres'] > 0:
                slim_T = np.concatenate(slim_T)
                save_matrix_as_text( "%s-sep%d-em%d-slim.topic.vec" %( basename, slim_T.shape[0], topicvec.MAX_EM_ITERS ), 
                                            "slim topics", slim_T )
            