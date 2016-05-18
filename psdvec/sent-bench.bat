@echo off
rem cd d:\corpus
rem python corpus2liblinear.py -d aclImdb\test\pos -o sent-test.txt 1
rem python corpus2liblinear.py -d aclImdb\test\neg -o sent-test.txt -1 -a
rem python corpus2liblinear.py -d aclImdb\train\pos -o sent-train.txt 1
rem python corpus2liblinear.py -d aclImdb\train\neg -o sent-train.txt -1 -a
pushd d:\liblinear-2.1\windows
train -s7 -v10 \corpus\sent-train-PSD-reg.txt PSD-reg.model
predict \corpus\sent-test-PSD-reg.txt PSD-reg.model pred-output.txt
popd

