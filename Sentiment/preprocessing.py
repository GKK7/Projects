import nltk

nltk.download('stopwords')

# token is a singe unit of information that we would feed into a model in NLP

from nltk.corpus import stopwords

tweet = """I’m amazed how often in practice, not only does a @huggingface NLP model solve your problem, but one of their public finetuned checkpoints, is good enough for the job.

Both impressed, and a little disappointed how rarely I get to actually train a model that matters :("""
stop_words = stopwords.words('english')
print(stop_words[:10])

# stopwords are words that aren't particularly expressive and have less impact

stop_words = set(stop_words)

tweet = tweet.lower().split()

tweet_no_stopwords = [word for word in tweet if word not in stop_words]

print(" ".join(tweet))

print(" ".join(tweet_no_stopwords))

# replace users with <USER> and URLs with <URL> to not confuse the models

# special model tokens for BERT:
# [PAD] - same length sequences - can't mismatch dimensions 512 tokens
# [UNK] - unknown word to BERT
# [CLS] - appears at start of every sentence
# [SEP] - separator or end of sequence
# [MASK] - masking tokens to train with MLM - masked language modelling

# Stemming - simplify text before fed into a model
# Inflection - slight modification of a word

words_to_stem = ['happy', 'happiest', 'happier', 'cactus', 'cactii', 'elephant', 'elephants', 'amazed', 'amazing',
                 'amazingly', 'cement', 'owed', 'maximum']

from nltk.stem import PorterStemmer, LancasterStemmer

porter = PorterStemmer()
lancaster = LancasterStemmer()

stemmed = [(porter.stem(word), lancaster.stem(word)) for word in words_to_stem]

print("Porter | Lancaster")
for stem in stemmed:
    print(f"{stem[0]} | {stem[1]}")

# Lemmatization - reduces inflections down their real word root called a Lemmer

words = ['amaze', 'amazed', 'amazing']

import nltk

nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print([lemmatizer.lemmatize(word) for word in words])

from nltk.corpus import wordnet

print([lemmatizer.lemmatize(word, wordnet.VERB) for word in words])

# Canonical equivalents deals with characters that are identical when rendered, but in reality are completely different characters

# Compatibility equivalence - style changes on similar 1/2= 1+/+2

# Unicode Normalization - Composition and Decomposition
# Complex characters can be decomposed further into Unicode characters

import unicodedata

c_with_cedilla = "\u00C7"
print(c_with_cedilla)

c_plus_cedilla = "\u0043\u0327"  # \u0043 = Latin capital C, \u0327 = 'combining cedilla' (two characters)
print(c_plus_cedilla)
print(c_with_cedilla == c_plus_cedilla)

# NFD decomposition - the C is the same in both instances so it returns True
print(unicodedata.normalize('NFD', c_with_cedilla) == c_plus_cedilla)

# NFC decompose them into smaller components and compose them back
print(unicodedata.normalize('NFC', c_with_cedilla) == c_plus_cedilla)

# The NFK encodings do not decompose characters into smaller components, they decompose characters into their normal versions.
print(unicodedata.normalize('NFKD', 'ℌ'))

print(unicodedata.normalize('NFKC', "\u210B\u0327"), unicodedata.normalize('NFKD', "\u210B\u0327"))

