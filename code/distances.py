import pickle
import faiss
import numpy as np
import sys
import pandas as pd

chemin_phrases = "/path"

def charger_index(chemin_index:str, nombre_phrases:int):
    index_vecteurs = faiss.read_index(chemin_index)
    vecteurs = index_vecteurs.reconstruct_batch(range(0, nombre_phrases))
    D, I = index_vecteurs.search(vecteurs[:], nombre_phrases)
    return D, I

def ordonner_index(distances:np.ndarray, indices:np.ndarray):
    indices_plus_similaires = indices[:,1].argsort()[::-1]
    indices_ordonnés, distances_ordonnées = (indices[indices_plus_similaires], \
                                             distances[indices_plus_similaires])
    return indices_ordonnés, distances_ordonnées

def créer_dictionnaire(distances:np.ndarray, indices:np.ndarray, phrases:list):
    """
    Docstring for créer_dictionnaire
    
    :param distances: Description
    :type distances: np.ndarray
    :param indices: Description
    :type indices: np.ndarray
    :param phrases: Description
    :type phrases: list
    """
    pass

def main():
    with open(chemin_phrases, "r") as t:
        phrases = pickle.load(t)
    n = len(phrases)




