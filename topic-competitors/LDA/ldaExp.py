import sys
import pdb
import os
import getopt
import time
import gensim
from sklearn.datasets import load_svmlight_file
from scipy.sparse import csr_matrix
from corpusLoader import *

# 从语料库名映射到加载函数，之后调用
corpus2loader = { '20news': load_20news, 'reuters': load_reuters }

def usage():
    print """Usage: ldaExp.py corpus_name"""

corpusName = sys.argv[1]
# 加载函数
loader = corpus2loader[corpusName]

# 20news的文档数和类别数都多些，所以主题数设大一些
if corpusName == "20news":
    topicNum = 100
else:
    topicNum = 50

# 两个语料都已分成train和test集合。后面分别处理
setNames = [ 'train', 'test' ]
basenames = []
subcorpora = []
corpus = []
word2id = {}
id2word = {}
maxWID = 0

for setName in setNames:
    print "Process set '%s':" %setName
    
    # 加载语料的train或test子集，单词以句子为单位放入 orig_docs_words，类比放在 orig_docs_cat
    setDocNum, orig_docs_words, orig_docs_name, orig_docs_cat, cats_docsWords, \
            cats_docNames, category_names = loader(setName)
    # 文件名前缀
    basename = "%s-%s-%d" %( corpusName, setName, setDocNum )
    basenames.append(basename)
    
    # 当前循环所处理的语料子集，是一个list的list。每个外层list元素对应一个文档
    # 每个内层list为一串 (word_id, frequency) 的pair
    # 这种格式是gensim的标准输入格式
    subcorpus = []
    
    # 保存原始文本，以供人查看
    orig_filename = "%s.orig.txt" %basename
    ORIG = open( orig_filename, "w" )

    # 每个 wordsInSentences 对应一个文档
    # 每个 wordsInSentences 由许多句子组成，每个句子是一个list of words
    for wordsInSentences in orig_docs_words:
        # 统计当前文档的每个词的频率
        doc_wid2freq = {}
        # 循环取当前文档的一个句子
        for sentence in wordsInSentences:
            for w in sentence:
                w = w.lower()
                ORIG.write( "%s " %w )
                
                # 如果w已在word2id映射表中，映射成wid
                if w in word2id:
                    wid = word2id[w]
                # 否则，把w加入映射表，并映射成新wid
                else:
                    wid = maxWID
                    word2id[w] = maxWID
                    id2word[maxWID] = w
                    maxWID += 1
                
                # 统计 wid 的频率
                if wid in doc_wid2freq:
                    doc_wid2freq[wid] += 1
                else:
                    doc_wid2freq[wid] = 1
                    
        ORIG.write("\n")
        # 把文档中出现的wid按id大小排序
        sorted_wids = sorted( doc_wid2freq.keys() )
        doc_pairs = []
        # 把 (wid, frequency) 的对追加到当前文档的list中
        for wid in sorted_wids:
            doc_pairs.append( (wid, doc_wid2freq[wid]) )
        
        # 当前文档的list已经完全生成，把它加入subcorpus，即语料子集的list中    
        subcorpus.append(doc_pairs)

    ORIG.close()
    print "%d original docs saved in '%s'" %( setDocNum, orig_filename )

    # 把整个语料子集list与之前的list合并，得到一个包含train和test集合的所有文档的集合
    corpus += subcorpus
    # 这里把train和test集合分开放，之后会把不同集合的每个文档的“doc-topic比例”保存成不同文件
    subcorpora.append( (subcorpus, orig_docs_cat) )
    
print "Training LDA..."
startTime = time.time()
# LDA训练的时候是把train和test放一起训练的(更严格的办法应该是只用train集合来训练)
lda = gensim.models.ldamodel.LdaModel( corpus=corpus, num_topics=topicNum, passes=20 )
endTime = time.time()
print "Finished in %.1f seconds" %( endTime - startTime )

for i in xrange(2):
    lda_filename = "%s.svm-lda.txt" %basenames[i]
    LDA = open( lda_filename, "w" )
    print "Saving topic proportions into '%s'..." %lda_filename
    
    # 拿出一个语料子集 (train或者test)
    subcorpus, labels = subcorpora[i]

    # 遍历子集中每个文档
    for d, doc_pairs in enumerate(subcorpus):
        label = labels[d]
        # 把当前文档作为输入，用训练好的LDA模型求“doc-topic比例”
        topic_props = lda.get_document_topics( doc_pairs, minimum_probability=0.001 )
        LDA.write( "%d" %label )
        # 把K个比例保存成K个特征，svmlight格式
        for k, prop in topic_props:
            LDA.write(" %d:%.3f" %(k, prop) )
        LDA.write("\n")
    LDA.close()
    print "%d docs saved" %len(subcorpus)
