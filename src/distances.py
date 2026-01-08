import pickle
import faiss
import numpy as np
import os
import sys
import pandas as pd
from tqdm import tqdm

chemin_phrases = "results/corpus_en_phrases.pickle"
chemin_index = "results/vecteurs.index"
chemin_resultats = "results/"

def charger_index(chemin_index:str, nombre_phrases:int):
    index_vecteurs = faiss.read_index(chemin_index)
    vecteurs = index_vecteurs.reconstruct_batch(range(0, nombre_phrases))
    D, I = index_vecteurs.search(vecteurs[:], nombre_phrases)
    print(I[:5, :5]) # sanity check
    return D, I

def ordonner_index(distances:np.ndarray, indices:np.ndarray):
    indices_plus_similaires = distances[:,1].argsort()[::-1]
    print("Ordonnancement de l'index...")
    indices_ordonnés, distances_ordonnées = (indices[indices_plus_similaires, :], distances[indices_plus_similaires, :])
    return indices_ordonnés, distances_ordonnées

def faire_table(indices_ordonnés:np.ndarray, distances_ordonnées:np.ndarray,\
                phrases:list, taille_bout=5000, nom="cosinus"):
    phrases_array = np.array(phrases)
    fichier = os.path.join(chemin_resultats, f"{nom}_voisins.csv")
    bar_progrès = tqdm(range(0, len(indices_ordonnés), taille_bout))
    portions = len(indices_ordonnés) // taille_bout
    for i, début in enumerate(bar_progrès):
        bar_progrès.set_description(f"En train de processer bout {i+1} sur {portions}...") 
        fin = min(début + taille_bout, len(indices_ordonnés))
        bout_indices = indices_ordonnés[début:fin]
        bout_distances = distances_ordonnées[début:fin]
        
        I0 = bout_indices[:, 0]
        I1 = bout_indices[:, 1]
        I2 = bout_indices[:, 2]
        I3 = bout_indices[:, 3]
        
        dictionnaire = {"I0": I0, "P0": phrases_array[I0],
                        "I1": I1, "P1": phrases_array[I1], "D1": bout_distances[:, 1],
                        "I2": I2, "P2": phrases_array[I2], "D2": bout_distances[:, 2],
                        "I3": I3, "P3": phrases_array[I3], "D3": bout_distances[:, 3]}
        
        df = pd.DataFrame(dictionnaire, index=I0)
        df.to_csv(fichier, mode='a', header=(début==0))

def maxima(phrases:np.array):
    max_longueurs = max([len(phrase) for phrase in phrases])
    max_distances = len(phrases) - 1
    return max_longueurs, max_distances

def ponderation(phrases, index_comparant, index_compare, similarite_cosinus, max_longueurs, max_distances):
    distance_score = abs(index_comparant - index_compare) / max_distances
    longueur_score = max(len(phrases[index_comparant]), len(phrases[index_compare])) / max_longueurs
    return 0.7 * similarite_cosinus + 0.2 * longueur_score + 0.1 * distance_score

def table_scores(phrases:np.array, distances:np.ndarray, indices:np.ndarray):
    matrice_scores = np.zeros_like(distances)
    n = len(phrases)
    max_longueurs, max_distances = maxima(phrases)
    for i in range(n):
        for j in range(n):
            if i != j:
                matrice_scores[i, j] = ponderation(phrases, i, indices[i,j], distances[i,j], max_longueurs, max_distances)
            else:
                matrice_scores[i, j] = 0.
    return matrice_scores

def main():
    with open(chemin_phrases, "rb") as t:
        phrases = pickle.load(t)
    print(f"Il y a {len(phrases)} phrases")
    d, i = charger_index(chemin_index, len(phrases))
    table_ponderee = table_scores(phrases, d, i)
    i_ordonnés, d_pond_ordonnees = ordonner_index(table_ponderee, i)
    i_ordonnés, d_ordonnées = ordonner_index(d, i)
    faire_table(i_ordonnés, d_ordonnées, phrases)
    faire_table(i_ordonnés, d_pond_ordonnees, phrases, nom="ponderes")

if __name__ == "__main__":
    main()



