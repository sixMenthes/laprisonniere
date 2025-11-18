

import spacy, re
import fr_dep_news_trf
from nltk.corpus import stopwords
import matplotlib.pyplot as plt




nlp = spacy.load("fr_dep_news_trf")

nlp = fr_dep_news_trf.load()
nlp.max_length = 1030000
#french_stopwords = stopwords.words('french')


# print([(w.lemma_) for w in doc])

with open('/Users/leo/stagear/corpus/laprison.txt', "r") as albertine:
    corpus = albertine.read()



corpus = re.sub(r"(\b\n)|(?<=,|:|;|\.|\?|!)\n", " ", corpus)
corpus = re.sub(r"\n{2,}", "\n", corpus)

print("Number of paragraphs = %s" % (len(corpus.split("\n"))) )

# for number, paragraph in enumerate(corpus.split("\n")):
#     print(f"{number}: {paragraph}")

doc_pre_processed = nlp(corpus)
paragraph_original = []
paragraph_lemma = []
corpus_original = list()
corpus_lemma = list()
word_freq = dict()

for word in doc_pre_processed:

    word_freq[word.lemma_] = word_freq.get(word.lemma_, 0) + 1
    
    if (word.text == '\n'):
        # paragraph_original.append(word.text)
        # paragraph_lemma.append(word.lemma_)
        corpus_original.append(' '.join(paragraph_original))
        corpus_lemma.append(' '.join(paragraph_lemma))
        paragraph_original = []
        paragraph_lemma = []
    else:
        paragraph_original.append(word.text)
        paragraph_lemma.append(word.lemma_)

print(corpus_original[:21])

# with open("/Users/leo/stagear/corpus/laprison_original.csv", "w", newline='') as laprison_original_csv:
#     wr = csv.writer(laprison_original_csv, delimiter='\n')
#     wr.writerow(corpus_original)

# with open("/Users/leo/stagear/corpus/laprison_lemma.csv", "w", newline='') as laprison_lemma_csv:
#     wr = csv.writer(laprison_lemma_csv, delimiter='\n')
#     wr.writerow(corpus_lemma)


# sorted_data = dict(sorted(word_freq.items(), key=lambda item: item[1], reverse=True))
# keys = list(sorted_data.keys())
# values = list(sorted_data.values())

# kneedle = KneeLocator(values, keys, S=100, curve="concave", direction="decreasing")
# kneedle.plot_knee_normalized()




"""
doc_pre_processed = nlp(corpus)






#by_sentence_prepro = list()
by_paragraph_original = list()
by_paragraph_lemma = list()
#sentence_prepro = list()
paragraph_original = []
paragraph_lemma = []
vocab = dict()
vocab_set = set()



#creates a list of sentences and the vocabulary
for word in doc_pre_processed:
    vocab[word.text] = vocab.get(word.text, 0) + 1
    
    if (word.text == '\n'):
        print("it works!")
        paragraph_original.append(word.text)
        paragraph_lemma.append(word.lemma_)
        by_paragraph_original.append(' '.join(paragraph_original))
        by_paragraph_lemma.append(' '.join(paragraph_lemma))
        sentence_original = []
        sentence_lemma = []
    else:
        paragraph_original.append(word.text)
        paragraph_lemma.append(word.lemma_)




# print(len(by_sentence_original))

with open("/Users/leo/stagear/corpus/laprison_original.csv", "w", newline='') as laprison_original_csv:
    wr = csv.writer(laprison_original_csv, delimiter='\n')
    wr.writerow(by_sentence_original)

with open("/Users/leo/stagear/corpus/laprison_lemma.csv", "w", newline='') as laprison_lemma_csv:
    wr = csv.writer(laprison_lemma_csv, delimiter='\n')
    wr.writerow(by_sentence_lemma)

#print(list(sorted(vocab.items(), key=lambda item: item[1])))
"""