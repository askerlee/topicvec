#!/bin/sh
export ROOT=/home/shaohua/D
#export CORPUS=$ROOT/corpus/cleanwiki.txt
#export DIM=500
#export MINCOUNT=100
#export SUFFIX=wiki
export CORPUS=$ROOT/corpus/rcv1clean.txt
export DIM=50
export MINCOUNT=50
export SUFFIX=rcv1

cd $ROOT/corpus/
echo PSD:
./fact-$SUFFIX.sh
cd $ROOT/word2vec
echo word2vec:
time ./word2vec -train $CORPUS -output $ROOT/corpus/word2vec-$SUFFIX.vec -size $DIM -window 5 -sample 1e-4 -negative 15 -min-count $MINCOUNT
cd $ROOT/corpus/glove/
echo glove:
time ./$SUFFIX.sh
cd $ROOT/corpus/singular/
echo singular:
time ./singular --corpus $CORPUS --output ./$SUFFIX --rare $MINCOUNT --window 3 --dim $DIM
echo PPM and SVD:
cd $ROOT/corpus/hyperwords
./train-$SUFFIX.sh
echo Sparse:
tail -n+2 $ROOT/corpus/word2vec-$SUFFIX.vec >  $ROOT/corpus/word2vec-$SUFFIX-headless.vec
cd $ROOT/corpus/sparse/
time ./sparse ../word2vec-$SUFFIX-headless.vec 5 0.5 1e-5 4 sparse-$SUFFIX.vec
