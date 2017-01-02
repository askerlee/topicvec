import numpy as np
from utils import *

embedding_arrays = np.load("25000-180000-500-BLK-8.0.vec.npy")
V, vocab, word2ID, skippedWords_whatever = embedding_arrays
model = VecModel(V, vocab, word2ID, vecNormalize=True)
w1, w2 = predict_ana(model, "fish", "water", "plant", "soil")
print w1, w2
w1, w2 = predict_ana(model, "player", "team", "student", "classroom")
print w1, w2
