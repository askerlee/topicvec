python topicExp.py -s reuters train
python topicExp.py -p reuters-train-5770-sep91-em100-last.topic.vec reuters train,test
python classEval.py reuters topicprop
python classEval.py reuters topic-wvavg
