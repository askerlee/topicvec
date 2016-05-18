from sklearn import metrics
import sys

def getScores( true_classes, pred_classes, average):
    precision = metrics.precision_score( true_classes, pred_classes, average=average )
    recall = metrics.recall_score( true_classes, pred_classes, average=average )
    f1 = metrics.f1_score( true_classes, pred_classes, average=average )
    accuracy = metrics.accuracy_score( true_classes, pred_classes )
    return precision, recall, f1, accuracy

true_labelfile = sys.argv[1]
pred_labelfile = sys.argv[2]

TRUE = open(true_labelfile)
PRED = open(pred_labelfile)

true_classes = []
pred_classes = []

for line in TRUE:
    line = line.strip()
    label = int(line)
    true_classes.append(label)

for line in PRED:
    line = line.strip()
    label = int(line)
    pred_classes.append(label)

print metrics.classification_report(true_classes, pred_classes, digits=3)

for average in ['micro', 'macro']:
    precision, recall, f1, acc = getScores( true_classes, pred_classes, average )
    print "Prec (%s average): %.3f, recall: %.3f, F1: %.3f, Acc: %.3f" %(  average, 
                        precision, recall, f1, acc )
