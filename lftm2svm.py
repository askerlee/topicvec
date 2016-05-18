import sys

lftm_file = sys.argv[1]

train_words_file = "reuters-train-5770.gibbslda-words.txt"
train_label_file = "reuters-train-5770.slda-label.txt"
test_words_file = "reuters-test-2255.gibbslda-words.txt"
test_label_file = "reuters-test-2255.slda-label.txt"

LFTM_TOPIC = open(lftm_file)
TRAIN_WORDS = open(train_words_file)
TEST_WORDS = open(test_words_file)
TRAIN_LABELS = open(train_label_file)
TEST_LABELS = open(test_label_file)

for i in xrange(2):
	WORDS =  [ TRAIN_WORDS, TEST_WORDS ][i]
	LABELS = [TRAIN_LABELS, TEST_LABELS][i]
	if i == 0:
		output_file = "reuters-train-5770.svm-lftm.txt"
	else:
		output_file = "reuters-test-2255.svm-lftm.txt"

	OUTPUT = open(output_file, "w")
	
	setName = ["train", "test"][i]

	lineno = 0
	validDocNum = 0
	for line in WORDS:
		lineno += 1 
		line = line.strip()
		label_line = LABELS.readline().strip()
		if not line:
			print "Empty doc %s-%d skipped" %(setName, lineno)
			continue
		label = int(label_line)
		OUTPUT.write( "%d" %(label+1) )
		lftm_topic_line = LFTM_TOPIC.readline().strip()
		lftm_topicprops = lftm_topic_line.split(" ")
		for k in xrange(50):
			topicprop = float(lftm_topicprops[k])
			OUTPUT.write( " %d:%.3f" %(k+1, topicprop) )
		OUTPUT.write("\n")
		validDocNum += 1
	print "%d %s docs, %d written into '%s'" %(lineno, setName, validDocNum, output_file)
	OUTPUT.close()

lineno = 0
for line in LFTM_TOPIC:
	lineno += 1

if lineno > 0:
	print "Warn: %d lines left in '%s'" %(lineno, lftm_file)
