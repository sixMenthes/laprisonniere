import pickle
import os

"""
Regex qui marche un peu:

(?<! [A-Z])(\.{1,3}|!+|\?+|;|:)(?=( [A-Z]| «|\n|» |»\n|--|\n--))

il faut éliminer la question des \n
par ailleurs, on se rend compte que la présence des majuscules à la suite de la ponctuation est décisive.

Extrait compliqué:
 Il avait à dire «vous», et le moins souvent possible «Monsieur»,
le plaisir de quelqu'un dont le père n'avait jamais employé, en
s'adressant à mes parents, que la «troisième personne». Presque toutes
les photographies portaient une dédicace telle que: «A mon meilleur
ami». Une actrice plus ingrate et plus avisée avait écrit: «Au meilleur
des amis», ce qui lui permettait, m'a-t-on assuré, de dire que mon oncle
n'était nullement, et à beaucoup près, son meilleur ami, mais l'ami qui
lui avait rendu le plus de petits services, l'ami dont elle se servait,
un excellent homme, presque une vieille bête. 

Résultat du progamme ci-dessous:

{'’', '-', '=', '.', '^', '[', '»', ')', '%', '\ufeff', ',', ' ', ';', '_', ':', '*', '«', '!', '(', '/', "'", ']', '\n', '?', '°', '—', '\x7f'}

Le pourcentage n'apparaît qu'une fois, les astérisques sont liés aux changements de parties, l'égal a l'air d'être une erreur, les caractères qui commencent par \ restent à explorer, de même que les parenthèses. 

"""

chemin_corpus = os.path.abspath('./corpus_echos')
chemin_resultats = os.path.abspath('./corpus_echos')

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
    print(ponctuation)

if __name__ == "__main__":
    main()

    