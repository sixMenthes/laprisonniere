import os
import re
from tqdm import tqdm
import pickle
import spacy
from spacy.tokens import DocBin

nlp = spacy.load("fr_dep_news_trf")
proust_path = os.path.abspath('../corpus_echos/')
resultats_path = os.path.abspath('../results/')

def préprocesser_corpus(texte:str):
    texte = re.sub(r'\n', ' ', texte)
    texte = re.sub(r' +', ' ', texte)
    return texte

def phraser_doc(spacy_doc):
    phrases_tokénisées = []
    phrases_spacy = DocBin()
    for p in spacy_doc.sents:
        phrase = p.text.strip()
        if len(phrase) > 1 and any(c.isalnum() for c in phrase):
            phrases_spacy.add(p.as_doc())
            phrases_tokénisées.append(phrase)
    return phrases_spacy, phrases_tokénisées
        

def phraser_corpus(corpus_path):
    recherche_parsée = DocBin()
    recherche_texte = []
    oeuvres = sorted(os.listdir(corpus_path))
    bar_de_progrès = tqdm(oeuvres)
    for oeuvre in bar_de_progrès:
        bar_de_progrès.set_description(f"En train de parser: {oeuvre} ")
        oeuvre_path = os.path.join(corpus_path, oeuvre)
        with open(oeuvre_path, "r") as t:
            texte = préprocesser_corpus(t.read())
            doc = nlp(texte)
            s, p = phraser_doc(doc)
            recherche_parsée.merge(s)
            recherche_texte.extend(p)
    return recherche_parsée, recherche_texte

def main():
    doc_path = os.path.join(resultats_path, 'recherche_parsée.spacy')
    cornichon_path = os.path.join(resultats_path, 'corpus.pickle')
    doc_bin, phrases = phraser_corpus(proust_path)
    doc_bin.to_disk(doc_path)
    with open(cornichon_path, 'wb') as f:
        pickle.dump(phrases, f)
    


if __name__ == "__main__":
    main()




