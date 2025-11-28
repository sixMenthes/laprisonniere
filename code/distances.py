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
    print("Ordonnancement de l'index...")
    indices_ordonnés, distances_ordonnées = (indices[indices_plus_similaires, :], distances[indices_plus_similaires, :])
    return indices_ordonnés, distances_ordonnées

def faire_table(indices_ordonnés:np.ndarray, distances_ordonnées:np.ndarray,\
                phrases:list, taille_bout=5000):
    phrases_array = np.array(phrases)
    fichier = os.path.join(chemin_resultats, "voisins.csv")
    
    for début in range(0, len(indices_ordonnés), taille_bout):
        fin = min(début + taille_bout, len(indices_ordonnés))
        bout_indices = indices_ordonnés[début:fin]
        bout_distances = distances_ordonnées[début:fin]
        
        I0 = bout_indices[:, 0]
        I1 = bout_indices[:, 1]
        I2 = bout_indices[:, 2]
        I3 = bout_indices[:, 3]
        
        dictionnaire = {"P0": phrases_array[I0],
                        "P1": phrases_array[I1], "D1": bout_distances[:, 1],
                        "P2": phrases_array[I2], "D2": bout_distances[:, 2],
                        "P3": phrases_array[I3], "D3": bout_distances[:, 3]}
        df = pd.DataFrame(dictionnaire, index=I0)
        
        df.to_csv(fichier, mode='a', header=(début==0))

def main():
    with open(chemin_phrases, "rb") as t:
        phrases = pickle.load(t)
    d, i = charger_index(chemin_index, len(phrases))
    i_ordonnés, d_ordonnées = ordonner_index(d, i)
    faire_table(i_ordonnés, d_ordonnées, phrases)

if __name__ == "__main__":
    main()



