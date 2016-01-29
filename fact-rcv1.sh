#!/bin/bash
N0=50
time python factorize.py -w 15000 -n $N0 -E5 top2grams-rcv1.txt
time python factorize.py -v 15000-$N0-EM.vec  -n $N0 -t2 top2grams-rcv1.txt
