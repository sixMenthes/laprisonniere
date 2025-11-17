import numpy as np
import torch
import faiss
from transformers import AutoTokenizer, AutoModel
import pickle

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def créer_index_faiss(embeddings):
    d = embeddings.size(1)
    index = faiss.IndexFlatL2(d)
    embeddings_np = embeddings.cpu().numpy().astype('float32')
    index.add(embeddings)
    faiss.write_index(index, "../corpus/whole_proust/corpus_embeddings.index")



if __name__ == "__main__":
    print(f"Loading tokenizers etc.")
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    model = AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    d = 384

    with open('../corpus/whole_proust/corpus.pickle', 'rb') as f:
        phrases = pickle.load(f)
    print(f"Corpus loaded. Tokenizing corpus.")
    encoded_input = tokenizer(phrases, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    print(f"Calculating embeddings.")
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    print(f"Creating vector database.")
    créer_index_faiss(sentence_embeddings)