import os
import torch
import torch.nn.functional as F
os.environ['OMP_NUM_THREADS'] = '16'  
os.environ['OMP_WAIT_POLICY'] = 'PASSIVE'
import faiss
torch.set_num_threads(16)  # Match physical cores
torch.set_num_interop_threads(16)
from transformers import AutoTokenizer, AutoModel
import pickle
from tqdm import tqdm

resultats_chemin = os.path.abspath('./tests/phraseur/resultats')
tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3")
model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
task = 'text-matching'
task_id = model._adaptation_map[task]


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )

def encoder_par_batch(liste_corpus, n):

    tous_embeddings = []
    for i in tqdm(range(0, len(liste_corpus), n), desc="Encodage par batch"):
        phrases = liste_corpus[i:i + n]
        encoded_input = tokenizer(phrases, padding=True, truncation=True, return_tensors='pt')
        adapter_mask = torch.full((len(phrases),), task_id, dtype=torch.int32)
        with torch.no_grad():
            model_output = model(**encoded_input, adapter_mask=adapter_mask)
        embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        tous_embeddings.append(embeddings)
    tous_embeddings = torch.cat(tous_embeddings, dim=0)
    return tous_embeddings

def créer_index_faiss(embeddings, index_path):
    d = embeddings.size(1)
    index = faiss.IndexFlatIP(d) 
    embeddings_np = embeddings.cpu().numpy().astype('float32')
    index.add(embeddings_np)
    faiss.write_index(index, index_path)


def main():
    cornichon = os.path.join(resultats_chemin, 'corpus_en_phrases.pickle')
    indice_chemin = os.path.join(resultats_chemin, 'vecteurs.index')
    with open(cornichon, 'rb') as f:
        phrases = pickle.load(f)
    sentence_embeddings = encoder_par_batch(phrases, 64)
    créer_index_faiss(sentence_embeddings, indice_chemin)


if __name__ == "__main__":
    main()