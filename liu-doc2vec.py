from utils import *
import pdb

def genDocEmbedding( setName, words_file, topics_file, label_file, V, word2ID, T ):
    WORDS = open(words_file)
    TOPICS = open(topics_file)
    LABEL = open(label_file)

    filename_trunk = words_file.split('.')[0]
    docvec_file = ".".join( [ filename_trunk, "svm-liu", "txt" ] )
    docvecbow_file = ".".join( [ filename_trunk, "svm-liubow", "txt" ] )

    DOCVEC = open( docvec_file, "w" )
    DOCVECBOW = open( docvecbow_file, "w" )

    dim = V.shape[1] + T.shape[1]

    lineno = 0
    emptyDocIds = []

    for word_line in WORDS:
        lineno += 1
        word_line = word_line.strip()
        topic_line = TOPICS.readline().strip()
        label_line = LABEL.readline().strip()
        # encounter an empty doc
        if not word_line:
            words = []
            topics = []
        else:
            words = word_line.split(" ")
            topics = topic_line.split(" ")
        assert len(words) == len(topics), \
            "Words number %d != topic number %d in line %d" %( len(words), len(topics), lineno )
        label = int(label_line)

        sum_vec = np.zeros(dim)
        doc_vec = np.zeros(dim)
        validWordNum = 0

        wid2freq = {}

        for i in xrange(len(words)):
            word = words[i]
            topic = int(topics[i])

            if word not in word2ID:
                continue
            validWordNum += 1
            wid = word2ID[word]
            sum_vec += np.concatenate( [ V[wid], T[topic] ] )

            if wid in wid2freq:
                wid2freq[wid] += 1
            else:
                wid2freq[wid] = 1

        if validWordNum > 0:
            doc_vec = sum_vec / validWordNum
        else:
            emptyDocIds.append(lineno)

        sorted_wids = sorted( wid2freq.keys() )

        DOCVEC.write( "%d" %(label+1) )
        DOCVECBOW.write( "%d" %(label+1) )
        
        for k in xrange(dim):
            DOCVEC.write( " %d:%.3f" %( k + 1, doc_vec[k] ) )
            DOCVECBOW.write( " %d:%.3f" %( k + 1, doc_vec[k] ) )

        for wid in sorted_wids:
            # first dim indices are reserved for topic features, so add dim here
            # add 1 to make wid start from 1
            DOCVECBOW.write( " %d:%d" %( wid + dim + 1, wid2freq[wid] ) )

        DOCVEC.write("\n")
        DOCVECBOW.write("\n")

    print "%d %s docs converted to Liu et al's docvec in svm format." %( lineno, setName )
    if len(emptyDocIds) > 0:
        print "Empty docs: %s" %emptyDocIds

    DOCVEC.close()
    DOCVECBOW.close()
    WORDS.close()
    TOPICS.close()
    LABEL.close()

corpus = sys.argv[1]

if corpus == '20news':
    train_words_file = "20news-train-11314.gibbslda-words.txt"
    train_topics_file = "20news-train-11314.gibbslda-topics.txt"
    train_wordvec_file = "20news-train-11314.liu-wordvec2.txt"
    train_topicvec_file = "20news-train-11314.liu-topicvec2.txt"
    train_label_file = "20news-train-11314.slda-label.txt"
    test_words_file = "20news-test-7532.gibbslda-words.txt"
    test_topics_file = "20news-test-7532.gibbslda-topics.txt"
    test_label_file = "20news-test-7532.slda-label.txt"
else:
    train_words_file = "reuters-train-5770.gibbslda-words.txt"
    train_topics_file = "reuters-train-5770.gibbslda-topics.txt"
    train_wordvec_file = "reuters-train-5770.liu-wordvec2.txt"
    train_topicvec_file = "reuters-train-5770.liu-topicvec2.txt"
    train_label_file = "reuters-train-5770.slda-label.txt"
    test_words_file = "reuters-test-2255.gibbslda-words.txt"
    test_topics_file = "reuters-test-2255.gibbslda-topics.txt"
    test_label_file = "reuters-test-2255.slda-label.txt"

V, vocab, word2ID, skippedWords_whatever = load_embeddings(train_wordvec_file)
T = load_matrix_from_text( train_topicvec_file, "topic embedding" )
genDocEmbedding( "train", train_words_file, train_topics_file, train_label_file, V, word2ID, T )
genDocEmbedding( "test", test_words_file, test_topics_file, test_label_file, V, word2ID, T )
