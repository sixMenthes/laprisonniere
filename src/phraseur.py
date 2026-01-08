import os
import regex as re
from collections import defaultdict
from tqdm import tqdm
import pickle
import spacy
from spacy.tokens import DocBin

séparateur = re.compile(r'(?<=[^A-Z](?:(?:(?<= \. \.|[^\.]) \.)| !| \?)) (?=[A-Z]|\(|\)|«|-(?= -))|(?<=;) |(?<=») (?=«)|(?<=(?<=(?:[!\?\.]) )») ')
nlp = spacy.load("fr_dep_news_trf")
proust_chemin = os.path.abspath('./corpus_echos')
resultats_chemin = os.path.abspath('./results')
dict_doublons = defaultdict(list)

def préprocesser_corpus(texte:str):
    texte = re.sub(r'\n', ' ', texte)
    texte = re.sub(r'[\x00-\x1F\x7F\uFEFF]', '', texte)
    texte = re.sub(r"(\*|\^|’|:|\/|»|;|\.|%|!|,|-|—|°|«|\)|\?|_|\[|'|\]|\(|=\
                    |\))", r" \1 ", texte)
    texte = re.sub(r' +', ' ', texte)
    return texte

def phraser_doc(phrases:list, titre:str):
    phrases_oeuvre = []
    phrases_parsées = DocBin()
    for p in phrases:
        phrase = p.strip()
        dict_doublons[phrase].append(titre)
        if len(phrase) > 1 and any(c.isalnum() for c in phrase) \
            and len(dict_doublons[phrase]) == 1:
            phrases_oeuvre.append(phrase)
    for p in nlp.pipe(phrases_oeuvre):
        phrases_parsées.add(p)
    
    return phrases_oeuvre, phrases_parsées

def phraser_corpus(corpus_path):
    recherche_texte = []
    recherche_parsée = DocBin()
    oeuvres = sorted(os.listdir(corpus_path))
    bar_de_progrès = tqdm(oeuvres)
    for oeuvre in bar_de_progrès:
        bar_de_progrès.set_description(f"En train de parser {oeuvre}")
        oeuvre_chemin = os.path.join(corpus_path, oeuvre)
        with open(oeuvre_chemin, "r") as t:
            texte = préprocesser_corpus(t.read())
            phrases = re.split(séparateur,texte)  # segmentation happens here
            s, p = phraser_doc(phrases, oeuvre) # first review
            recherche_texte.extend(s)
            recherche_parsée.merge(p)
    return recherche_texte, recherche_parsée

def main():
    doc_chemin = os.path.join(resultats_chemin, 'recherche_parsée.spacy')
    cornichon_phrases = os.path.join(resultats_chemin, 'corpus_en_phrases.pickle')
    cornichon_doublons = os.path.join(resultats_chemin, 'doublons.pickle')

    phrases, doc_bin = phraser_corpus(proust_chemin)
    doc_bin.to_disk(doc_chemin)

    with open(cornichon_phrases, 'wb') as f:
        pickle.dump(phrases, f)
    
    with open(cornichon_doublons, "wb") as f:
        pickle.dump(dict_doublons, f)
    


if __name__ == "__main__":
    main()




