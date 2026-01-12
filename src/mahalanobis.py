from src.phraseur import Corpus
from src.vectoriseur import Vectoriseur
import torch
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"CUDA disponible: {torch.cuda.is_available()}")

def transformation_mahalanobis(echantillon):
    echantillon = Vectoriseur(echantillon, dimensions=32).encoder_par_batch().to(device)
    with torch.no_grad():
        moyenne = echantillon.mean(0)
        echantillon = echantillon - moyenne
        cov = torch.mm(echantillon.t(), echantillon) / echantillon.size(0)
        L = torch.linalg.cholesky(cov)
        inv = torch.linalg.inv(L)
    return inv, moyenne

def mapper_vecteurs(corpus, inverse_covariance, moyenne):
    corpus = Vectoriseur(corpus, dimensions=32).encoder_par_batch().to(device)
    with torch.no_grad():
        corpus = torch.mm(corpus, inverse_covariance.t())
        corpus = corpus - moyenne 
        corpus = torch.linalg.norm(corpus, dim=1)
    return corpus.cpu()

def construire_csv(normes, corpus:Corpus):
    df = pd.DataFrame({
        "Distance mahalanobis": normes,
        "Phrase": corpus.texte,
        "Index phrase": range(len(corpus.texte)),
        "Roman": corpus.repères
        })
    df = df.sort_values("Distance mahalanobis")
    df.to_csv("results/normes_proximite.csv", index=False, encoding='utf-8')


def main():
    echantillon = Corpus("phrases_son.txt", style="echantillon")
    inv, moyenne = transformation_mahalanobis(echantillon.texte)
    corpus = Corpus("corpus_echos/", style="corpus")
    normes = mapper_vecteurs(corpus.texte, inv, moyenne).numpy()
    construire_csv(normes, corpus)


if __name__ == "__main__":
    main()









