import matplotlib.pyplot as plt
import numpy as np

def ordonner_matrice(distance_matrice, indices_matrice):
    print(f"Ré-ordonnant matrice...")
    matrice_ordonnee = np.zeros_like(distance_matrice)
    n = distance_matrice.shape[0]
    for i in range(n):
        for j in range(n):
            indice_objectif = indices_matrice[i, j]
            matrice_ordonnee[i, indice_objectif] = distance_matrice[i, j]

def plot_distance_heatmap(score_matrice, name="heatmap", figsize=(12, 10)):
    
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
    plt.savefig('distance_heatmap.png', dpi=300, bbox_inches='tight')
    
    return fig, ax


#if __name__ == "__main__":
#
## Load the npz file
#    fichier = "results/matrices.npz"
#    data = np.load(fichier)
#
## Extract the arrays
#    D = data['d']  # distance matrix
#    I = data['i']  # indices matrix
#
#    fig, ax = plot_distance_heatmap(D, I, name="Cosine Distances")
#    plt.savefig('distance_heatmap.png', dpi=300, bbox_inches='tight')