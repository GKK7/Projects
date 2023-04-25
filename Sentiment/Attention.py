hello = [0.71, 0.14, 0.51]
hi = [0.69, 0.15, 0.48]
tomato = [0.16, 0.59, 0.49]

import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(projection='3d')

xs = [hello[0], hi[0], tomato[0]]
ys = [hello[1], hi[1], tomato[1]]
zs = [hello[2], hi[2], tomato[2]]

ax.scatter(xs, ys, zs, s=100)

ax.set_xlim((0, 1))
ax.set_ylim((0, 1))
ax.set_zlim((0, 1))

# plt.show()

a = hello = [0.71, 0.14, 0.51]
b = hi = [0.69, 0.15, 0.48]
c = tomato = [0.16, 0.59, 0.49]

import numpy as np

a = np.array(hello)
b = np.array(hi)
c = np.array(tomato)

print(np.matmul(a, b.T))

print(np.matmul(a, c.T))

print(np.matmul(b, c.T))

# iclude a PAD padding vector

# BERT - Biderectional encoder repentations from transformers

# Rather than calculate the attention once we do it several times
# Multihead attention allows us to build several representation of attention between words
# 3 times attention = 3 attention heads.
# Key, Value and Query tensors but we want 1 attention tensor in the end so we concatenate them

# Sentiment Analysis

# 1. text is tokenized - each word
# 2. Token IDs fed into the model
# 3. First layer is an embedding layer and output a set of word vectors
# 4. From a simple number to a word vector is through and embedding array with each representing a vector
# 5. Model outputs a set of model activations into our transformer head
# 6. hidden state passed along to our head which consists of a linear layer
# 7. softmax function gives us the probabilities of the sentiment
# 8. Then we take the argmax - looks at these values and takes the maximum
# 9. The argmax depending on the position give us the predicted sentiment class

import flair
model = flair.models.TextClassifier.load('en-sentiment')

# Destillbert model - fed with a classification head that outputs positive or negative

text="I like you!!"
sentence=flair.data.Sentence(text)
print(sentence)
print(sentence.to_tokenized_string())
model.predict(sentence)
print(sentence)

print(sentence.get_labels()[0].score)

print(sentence.labels[0].score, sentence.labels[0].value)