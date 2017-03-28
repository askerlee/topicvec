python topicExp.py -s reuters train
python topicExp.py -i reuters-train-5770-sep91-em150-best.topic.vec reuters train,test
python classEval.py reuters topicprop
python classEval.py reuters topic-wvavg
