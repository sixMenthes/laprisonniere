import os

chemin_corpus = os.path.abspath('./corpus_echos')
chemin_resultats = os.path.abspath('./tests')

def extraire_caractères(oeuvre):
    return set(list(oeuvre))

def décerner_ponctuation(caractères: set):
    return set(c for c in caractères if not c.isalnum())

def main():
    ponctuation = set()
    oeuvres = os.listdir(chemin_corpus)
    for titre in oeuvres:
        chemin_oeuvre = os.path.join(chemin_corpus, titre)
        with open(chemin_oeuvre, 'r') as o:
            texte = o.read()
            ponctuation |= décerner_ponctuation(extraire_caractères(texte))
    
    texte_ponctuation = os.path.join(chemin_resultats, 'ponctuation.txt')
    with open(texte_ponctuation,"w") as f:
       for p in ponctuation:
           f.write(f"{ascii(p)}|") 

if __name__ == "__main__":
    main()

    