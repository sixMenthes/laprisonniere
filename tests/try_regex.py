import re
import os

destination = os.path.abspath("/Users/leo/M2TALS1/cv/github_ordering/laprisonniere/tests/corpus")

corpus_chemin = os.path.abspath("/Users/leo/M2TALS1/cv/github_ordering/laprisonniere/corpus_echos")

def préprocesser_corpus(texte:str):
    texte = re.sub(r'\n', ' ', texte)
    texte = re.sub(r'[\x00-\x1F\x7F\uFEFF]', '', texte)
    texte = re.sub(r"(\*|\^|’|:|\/|»|;|\.|%|!|,|-|—|°|«|\)|\?|_|\[|'|\]|\(|=\
                    |\))", r" \1 ", texte)
    texte = re.sub(r' +', ' ', texte)
    return texte

#def segmenter_corpus(texte:str):


def main():

    originaux = os.listdir(corpus_chemin)
    for original in originaux:
        nom = original
        oeuvre = os.path.join(corpus_chemin, original)
        target = os.path.join(destination, nom)
        with open(oeuvre, "r", encoding="UTF-8") as o:
            cote = o.read()
        with open(target, "w", encoding="UTF-8") as d:
            d.write(préprocesser_corpus(cote))

if __name__ == "__main__":
    main()