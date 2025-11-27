import os
import regex as re
import random

# random.seed(22)

séparateur = re.compile(r'(?<=[^A-Z](?:(?:(?<= \. \.|[^\.]) \.)| !| \?)) (?=[A-Z]|\(|\)|«|-(?= -))|(?<=;) |(?<=») (?=«)|(?<=(?<=(?:[!\?\.]) )») ')
origin = "tests/corpus/"
oeuvres = os.listdir(origin)
nombre_oeuvres = len(oeuvres)

def séparer(corpus):
    return re.split(séparateur, corpus)

def dés():
    n = random.randint(0,nombre_oeuvres-1)
    oeuvre = oeuvres[n]
    with open(os.path.join(origin, oeuvre), "r") as f:
        texte = f.read()
    liste = séparer(texte)
    début_phrases = random.randint(0, len(liste))
    for i in range(début_phrases, début_phrases+5):
        print(f"* {liste[i]}\n")

    print(f"-"*40)
    print(f"NAME: {oeuvre}")
    print(f"Nº of first: {début_phrases}")
    print(f"-"*40)
# if __name__=="__main__":
#    main()

