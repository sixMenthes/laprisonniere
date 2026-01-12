import matplotlib.pyplot as plt
import numpy as np

def ordonner_diachronique(distance_matrice, indices_matrice):
    print(f"Ré-ordonnant matrice...")
    matrice_ordonnee = np.zeros_like(distance_matrice)
    n = distance_matrice.shape[0]
    for i in range(n):
        for j in range(n):
            indice_objectif = indices_matrice[i, j]
            matrice_ordonnee[i, indice_objectif] = distance_matrice[i, j]

def plot_distance_heatmap(score_matrice, name="Cosinus", figsize=(12, 10)):
    
    tick_positions = [0, 5121, 11601, 19727, 27841, 33324, 36765]
    titre_romans = ["Du côté", "Jeunes filles", "Guermantes", "Sodome", 
                   "Prisonnière", "Albertine", "Le temps"]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(score_matrice, cmap='viridis', aspect='auto')
    
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Distance", rotation=-90, va="bottom")
    
    # Set ticks at volume boundaries
    ax.set_xticks(tick_positions, labels=titre_romans, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(tick_positions, labels=titre_romans)
    
    ax.set_title(f'Similitude entre les phrases dans La Recherche: {name}')
    
    plt.tight_layout()
    plt.savefig(f'results/distance_heatmap_{name}.png', dpi=300, bbox_inches='tight')
    
    return 0



