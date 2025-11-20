import re
import os

destination = os.path.abspath("./tests/exemple_ponct_spaced.txt")

origin = os.path.abspath("./tests/exemple_ponct.txt")

def préprocesser_corpus(texte:str):
    texte = re.sub(r'\n', ' ', texte)
    texte = re.sub(r'[\x00-\x1F\x7F\uFEFF]', '', texte)
    texte = re.sub(r"(\*|\^|’|:|\/|»|;|\.|%|!|,|-|—|°|«|\)|\?|_|\[|'|\]|\(|=\
                    |\))", r" \1 ", texte)
    texte = re.sub(r' +', ' ', texte)
    return texte

#def segmenter_corpus(texte:str):


def main():
    with open(origin, "r", encoding="UTF-8") as o:
        cote = o.read()
    with open(destination, "w", encoding="UTF-8") as d:
        d.write(préprocesser_corpus(cote))

if __name__ == "__main__":
    main()