import numpy as np
import os
import torch
os.environ['OMP_NUM_THREADS'] = '16'  
os.environ['OMP_WAIT_POLICY'] = 'PASSIVE'
import faiss
torch.set_num_threads(16)  # Match physical cores
torch.set_num_interop_threads(16)
from transformers import AutoTokenizer, AutoModel
import pickle
import tqdm

resultats_path = os.path.abspath('./results/')

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def créer_index_faiss(embeddings, index_path):
    d = embeddings.size(1)
    index = faiss.IndexFlatL2(d)
    embeddings_np = embeddings.cpu().numpy().astype('float32')
    index.add(embeddings_np)
    faiss.write_index(index, index_path)

def encoder_par_batch(liste_corpus, n):

    print(f"Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    model = AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    all_embeddings = []

    for i in tqdm(range(0, len(liste_corpus), n), desc="Encodage par batch"):
        phrases = liste_corpus[i:i + n]

        encoded_input = tokenizer(phrases, padding=True, truncation=True, return_tensors='pt')

        with torch.no_grad():
            model_output = model(**encoded_input)
        
        all_embeddings.append(mean_pooling(model_output, encoded_input['attention_mask']))
    sentence_embeddings = torch.cat(all_embeddings, dim=0)

    return sentence_embeddings
    
def main():
    cornichon = os.path.join(resultats_path, 'corpus.pickle')
    indice_chemin = os.path.join(resultats_path, 'vecteurs.index')
    with open(cornichon, 'rb') as f:
        phrases = pickle.load(f)
    sentence_embeddings = encoder_par_batch(phrases, 64)
    créer_index_faiss(sentence_embeddings, indice_chemin)

if __name__ == "__main__":
    main()