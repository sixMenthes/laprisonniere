import os
import regex as re
from collections import defaultdict
from tqdm import tqdm
import pickle
import unicodedata

class Corpus:
    def __init__(self, chemin_corpus, chemin_sauvegarde="", style="corpus"):
        
        self.chemin_corpus = chemin_corpus
        self.chemin_sauvegarde = chemin_sauvegarde
        self.texte = []
        self.repères = []
        if style == "corpus":
            self.__phraser_corpus__()
        elif style == "echantillon":
            self.__phraser_échantillon__()

    def __préprocesser__(self, texte:str):
        texte = unicodedata.normalize('NFC', texte)
        texte = re.sub(r'\n', ' ', texte)
        texte = re.sub(r'[\x00-\x1F\x7F\uFEFF]', '', texte)
        texte = re.sub(r"(\*|\^|’|:|\/|»|;|\.|%|!|,|-|—|°|«|\)|\?|_|\[|'|\]|\(|=\
                        |\))", r" \1 ", texte)
        texte = re.sub(r' +', ' ', texte)
        return texte

    def __phraser_corpus__(self):
        séparateur = re.compile(r'(?<=[^A-Z](?:(?:(?<= \. \.|[^\.]) \.)| !| \?)) (?=[A-ZÁÀÂÉÈÊÔÎ]|\(|\)|«|-(?= -))|(?<=;) |(?<=») (?=«)|(?<=(?<=(?:[!\?\.]) )») ')

        oeuvres = sorted(os.listdir(self.chemin_corpus))
        for oeuvre in oeuvres:
            with open(os.path.join(self.chemin_corpus, oeuvre), "r") as t:
                texte = t.read()
                texte = self.__préprocesser__(texte)
                phrases = re.split(séparateur,texte)  # segmentation happens here
                self.repères.extend([oeuvre] * len(phrases))
                self.texte.extend(phrases)

    def __phraser_échantillon__(self):
        with open(self.chemin_corpus, "r") as t:
            echantillon = t.read().split('\n\n')
            for phrase in echantillon:
                phrase, _, doc = re.split(r"( |\n)+(?=[0-9].+\.txt)", phrase)
                self.texte.append(self.__préprocesser__(phrase))
                self.repères.append(doc.strip())
            
    def __len__(self):
        return len(self.texte)
    
    def indiquer_volume(self, index):
        return self.repères[index]

    def sauvegarder(self):
        if self.chemin_sauvegarde == "":
            print(f"Veuillez entrer un chemin de sauvegarde.")
            return
        try:
            fichier_phrases = os.path.join(self.chemin_sauvegarde, "corpus_phrases.pickle")
            fichier_reperes = os.path.join(self.chemin_sauvegarde, "corpus_reperes.pickle")

            with open(fichier_phrases, "wb") as f:
                pickle.dump(self.corpus, f)
            with open(fichier_reperes, "wb") as f:
                pickle.dump(self.repères, f)

        except Exception as e:
            print(f"Erreur lors de la sauvegarde: {e}") 