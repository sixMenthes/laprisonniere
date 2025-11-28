import pickle
import faiss
import numpy as np
import os
import sys
import pandas as pd

chemin_phrases = "results/corpus_en_phrases.pickle"
chemin_index = "results/vecteurs.index"
chemin_resultats = "results/"

def charger_index(chemin_index:str, nombre_phrases:int):
    index_vecteurs = faiss.read_index(chemin_index)
    vecteurs = index_vecteurs.reconstruct_batch(range(0, nombre_phrases))
    D, I = index_vecteurs.search(vecteurs[:], nombre_phrases)
    print(I[:5, :5]) # sanity check
    fichier = os.path.join(chemin_resultats, "matrices.npz")
    with open(fichier, "wb") as f:
        np.savez_compressed(f, d=D, i=I)
    return D, I

def ordonner_index(distances:np.ndarray, indices:np.ndarray):
    indices_plus_similaires = distances[:,1].argsort()[::-1]
    indices_ordonnés, distances_ordonnées = (indices[indices_plus_similaires, :], \
                                             distances[indices_plus_similaires, :])
    return indices_ordonnés, distances_ordonnées

def faire_table(indices_ordonnés:np.ndarray, distances_ordonnées:np.ndarray,\
                phrases:list):
    I0 = indices_ordonnés[:, 0]
    I1 = indices_ordonnés[:, 1]
    I2 = indices_ordonnés[:, 2]
    I3 = indices_ordonnés[:, 3]
    dictionnaire = {"P1": phrases[I1], "D1": distances_ordonnées[:, 1], \
                    "P2": phrases[I2], "D2": distances_ordonnées[:, 2], \
                    "P3": phrases[I3], "D3": distances_ordonnées[:, 3]}
    df = pd.DataFrame(dictionnaire, index=phrases[I0])
    fichier = os.path.join(chemin_resultats, "voisins.csv")
    df.to_csv(fichier)

def main():
    with open(chemin_phrases, "rb") as t:
        phrases = pickle.load(t)
    n = len(phrases)
    d, i = charger_index(chemin_index, n)
    i_ordonnés, d_ordonnées = ordonner_index(d, i)
    faire_table(i_ordonnés, d_ordonnées, phrases)




