import os
import regex as re
from collections import defaultdict
from tqdm import tqdm
import pickle
import spacy
from spacy.tokens import DocBin

séparateur = re.compile(r'(?<=[^A-Z](?:(?:(?<= \. \.|[^\.]) \.)| !| \?)) (?=[A-Z]|\(|\)|«|-(?= -))|(?<=;) |(?<=») (?=«)|(?<=(?<=(?:[!\?\.]) )») ')
nlp = spacy.load("fr_dep_news_trf")
proust_path = os.path.abspath('./corpus_echos/')
resultats_path = os.path.abspath('./results/')
dict_doublons = defaultdict(list)

def préprocesser_corpus(texte:str):
    texte = re.sub(r'\n', ' ', texte)
    texte = re.sub(r'[\x00-\x1F\x7F\uFEFF]', '', texte)
    texte = re.sub(r"(\*|\^|’|:|\/|»|;|\.|%|!|,|-|—|°|«|\)|\?|_|\[|'|\]|\(|=\
                    |\))", r" \1 ", texte)
    texte = re.sub(r' +', ' ', texte)
    return texte

def phraser_doc(spacy_doc, titre_oeuvre):
    phrases_tokénisées = []
    phrases_spacy = DocBin()
    for p in spacy_doc.sents:
        phrase = p.text.strip()
        dict_doublons[phrase].append(titre_oeuvre)
        if len(phrase) > 1 and any(c.isalnum() for c in phrase) \
            and len(dict_doublons[phrase]) == 1:
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
            doc = nlp(texte)  # segmentation happens here
            s, p = phraser_doc(doc, oeuvre)
            recherche_parsée.merge(s)
            recherche_texte.extend(p)
    return recherche_parsée, recherche_texte

def main():
    doc_path = os.path.join(resultats_path, 'recherche_parsée.spacy')
    cornichon_phrases = os.path.join(resultats_path, 'corpus_en_phrases.pickle')
    cornichon_doublons = os.path.join(resultats_path, 'doublons.pickle')

    doc_bin, phrases = phraser_corpus(proust_path)
    doc_bin.to_disk(doc_path)

    with open(cornichon_phrases, 'wb') as f:
        pickle.dump(phrases, f)
    
    with open(cornichon_doublons, "wb") as f:
        pickle.dump(dict_doublons, f)
    


if __name__ == "__main__":
    main()




