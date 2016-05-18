from sklearn import svm, metrics
from sklearn.datasets import load_svmlight_file
from utils import *
import sys

def getScores( true_classes, pred_classes, average):
    precision = metrics.precision_score( true_classes, pred_classes, average=average )
    recall = metrics.recall_score( true_classes, pred_classes, average=average )
    f1 = metrics.f1_score( true_classes, pred_classes, average=average )
    accuracy = metrics.accuracy_score( true_classes, pred_classes )
    return precision, recall, f1, accuracy
    
"""
TopicProp_ITER = int(sys.argv[1])
topicNum = int(sys.argv[2])
train_file = "20news-train-11314-sep%d-em40-i%d.topic.prop" %(topicNum, TopicProp_ITER)
test_file = "20news-test-7532-sep%d-em40-i%d.topic.prop" %(topicNum, TopicProp_ITER)

train_features, train_docs_cat_name = load_matrix_from_text( train_file, "training proportion", "\t" )
test_features, test_docs_cat_name = load_matrix_from_text( test_file, "test proportion", "\t" )

true_train_classes = []
true_test_classes = []

for train_cat_name in train_docs_cat_name[0]:
    true_train_classes.append( int(train_cat_name) )
for test_cat_name in test_docs_cat_name[0]:
    true_test_classes.append( int(test_cat_name) )
"""

corpus = sys.argv[1]
filetype = sys.argv[2]
# selected feature dimensions can be specified in the last argument as:
# 1-400 (starting from 1)
if len(sys.argv) > 3:
    dims = sys.argv[3].split("-")
    dims[0] = int(dims[0]) - 1
    dims[1] = int(dims[1])
else:
    dims = None
        
if corpus == '20news':
    train_file = "20news-train-11314.svm-%s.txt" %filetype
    test_file = "20news-test-7532.svm-%s.txt" %filetype
else:
    train_file = "reuters-train-5770.svm-%s.txt" %filetype
    test_file = "reuters-test-2255.svm-%s.txt" %filetype

train_features_sparse, true_train_classes = load_svmlight_file(train_file)
test_features_sparse, true_test_classes = load_svmlight_file(test_file)
#nonzeroColIDs = np.union1d( train_features_sparse.nonzero()[1], test_features_sparse.nonzero()[1] )
#train_features = train_features_sparse[:, nonzeroColIDs].toarray()
#test_features = test_features_sparse[:, nonzeroColIDs].toarray()
    
#pdb.set_trace()
#print "%dx%d sparse feature matrices reduced to %dx%d" %( tuple(train_features_sparse.shape) +
#                                                tuple(train_features.shape) )

train_features = train_features_sparse.toarray()
test_features = test_features_sparse.toarray()

print "Train: %dx%d. Test: %dx%d" %( tuple( train_features.shape + test_features.shape ) )

if dims:
    train_features = train_features[ :, dims[0]:dims[1] ]
    test_features = test_features[ :, dims[0]:dims[1] ]
    print "Choose only features %d-%d" %( dims[0]+1, dims[1] )
else:
    train_features = train_features[ :, : ]
    test_features = test_features[ :, : ]
        
model = svm.LinearSVC(penalty='l1', dual=False)

print "Training...",
model.fit( train_features, true_train_classes )
print "Done."

pred_train_classes = model.predict( train_features )
pred_test_classes = model.predict( test_features )

print metrics.classification_report(true_train_classes, pred_train_classes, digits=3)
print metrics.classification_report(true_test_classes, pred_test_classes, digits=3)

for average in ['micro', 'macro']:
    train_precision, train_recall, train_f1, train_acc = getScores( true_train_classes, pred_train_classes, average )
    print "Train Prec (%s average): %.3f, recall: %.3f, F1: %.3f, Acc: %.3f" %( average, 
                        train_precision, train_recall, train_f1, train_acc )
    
    test_precision, test_recall, test_f1, test_acc = getScores( true_test_classes, pred_test_classes, average )
    print "Test Prec (%s average): %.3f, recall: %.3f, F1: %.3f, Acc: %.3f" %(  average, 
                        test_precision, test_recall, test_f1, test_acc )
