import os
import re
#from tqdm import tqdm
import pickle

separateur = re.compile(r'\.{1,3} |\.» |\: |\; |\! |\?')
proust_path = os.path.abspath('../corpus/whole_proust')

def tokenizer_corpus(corpus_path):
    tokenizées_phrases = []
    oeuvres = os.listdir(corpus_path)
    for oeuvre in oeuvres:
        oeuvre_path = os.path.join(corpus_path, oeuvre)
        with open(oeuvre_path, "r") as texte:
            texte = texte.read()
            texte = re.sub(r'\n', ' ', texte)                    
            oeuvre_en_liste = re.split(separateur, texte)
            tokenizées_phrases += oeuvre_en_liste

    return tokenizées_phrases

def sauvegarder_corpus(corpus_tokénizé):
    with open('../corpus/whole_proust/corpus.pickle', 'wb') as f:
        pickle.dump(corpus_tokénizé, f)

if __name__ == "__main__":
    phrases = tokenizer_corpus(proust_path)
    sauvegarder_corpus(phrases)





