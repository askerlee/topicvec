import gensim.models.doc2vec as doc2vec
import sys
import pdb

corpus = sys.argv[1]

if corpus == '20news':
    all_words_file = "20news-all-18791.gibbslda-bow.txt"
    train_label_file = "20news-train-11314.slda-label.txt"
    train_docvec_file = "20news-train-11314.svm-doc2vec.txt"
    test_label_file = "20news-test-7532.slda-label.txt"
    test_docvec_file = "20news-test-7532.svm-doc2vec.txt"
    all_count = 18791
    train_count = 11285
    test_count = 7506
else:
    all_words_file = "reuters-all-8025.gibbslda-bow.txt"
    train_label_file = "reuters-train-5770.slda-label.txt"
    train_docvec_file = "reuters-train-5770.svm-doc2vec.txt"
    test_label_file = "reuters-test-2255.slda-label.txt"
    test_docvec_file = "reuters-test-2255.svm-doc2vec.txt"
    all_count = 8025
    train_count = 5770
    test_count = 2255

dim = 400
corpus = doc2vec.TaggedLineDocument(all_words_file)
model = doc2vec.Doc2Vec(corpus,size=dim, window=8, min_count=5, workers=4)
TRAIN_DOC2VEC = open(train_docvec_file, "w")
TRAIN_LABEL = open(train_label_file)

#pdb.set_trace()

for d in xrange(1, train_count + 1):
    doc_vec = model.docvecs[d]
    label_line = TRAIN_LABEL.readline().strip()
    label = int(label_line)

    TRAIN_DOC2VEC.write( "%d" %(label+1) )

    for k in xrange(dim):
        TRAIN_DOC2VEC.write( " %d:%.3f" %( k + 1, doc_vec[k] ) )

    TRAIN_DOC2VEC.write("\n")

TRAIN_DOC2VEC.close()

print "%d doc vecs written in svm format into '%s'" %( train_count, train_docvec_file )

TEST_DOC2VEC = open(test_docvec_file, "w")
TEST_LABEL = open(test_label_file)
for d in xrange(train_count + 1, all_count + 1):
    doc_vec = model.docvecs[d]
    label_line = TEST_LABEL.readline().strip()
    label = int(label_line)

    TEST_DOC2VEC.write( "%d" %(label+1) )

    for k in xrange(dim):
        TEST_DOC2VEC.write( " %d:%.3f" %( k + 1, doc_vec[k] ) )

    TEST_DOC2VEC.write("\n")

TEST_DOC2VEC.close()

print "%d doc vecs written in svm format into '%s'" %( test_count, test_docvec_file )
