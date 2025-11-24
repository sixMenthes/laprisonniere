import regex as re

origin = "tests/corpus/0ducote.txt"
séparateur = re.compile(r'(?<=[^A-Z]((?<= \. \.|[^\.]) \.| !| \?)(?= [A-Z]| «| »| \()|;|\)) ')

with open(origin, "r") as f:
    texte = f.read()

def séparer(corpus):
    return re.split(séparateur, corpus)

