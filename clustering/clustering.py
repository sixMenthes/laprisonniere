#!/Users/leo/virtualenvs/earvive/bin/ python3



import sys
import matplotlib.pyplot as plt

sys.path.append('/Users/leo/stagear/clustering/PLSA/plsa')

from plsa import Corpus, Pipeline, Visualize
from plsa.pipeline import DEFAULT_PIPELINE
from plsa.algorithms import PLSA

csv_file = '/Users/leo/stagear/corpus/laprison_lemma.csv'

pipeline = Pipeline(*DEFAULT_PIPELINE)
pipeline

corpus = Corpus.from_csv(csv_file, pipeline)
corpus

n_topics = 10

plsa = PLSA(corpus, n_topics, True)
plsa

result = plsa.fit()
plsa

result.topic

result.word_given_topic[0][:10] 

visualize = Visualize(result)
visualize

fig, ax = plt.subplots()
_ = visualize.convergence(ax)
fig.tight_layout()

fig, ax = plt.subplots()
_ = visualize.topics_in_doc(0, ax)
fig.tight_layout()
