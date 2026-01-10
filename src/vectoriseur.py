import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import pickle
from tqdm import tqdm

resultats_chemin = os.path.abspath('./results/')

class Vectoriseur:
    def __init__(self, phrases:list, tokenizer="jinaai/jina-embeddings-v3", model="jinaai/jina-embeddings-v3", task="text-matching"):

        self.phrases = phrases
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"CUDA disponible: {torch.cuda.is_available()}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.model = AutoModel.from_pretrained(model).to(self.device)
        self.task_id = self.model._adaptation_map[task]


    def __mean_pooling__(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def encoder_par_batch(self, n=256):

        tous_embeddings = []
        for i in tqdm(range(0, len(self.phrases), n), desc="Encodage par batch"):
            phrases = self.phrases[i:i + n]
            encoded_input = self.tokenizer(phrases, padding=True, truncation=True, return_tensors='pt')
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
            adapter_mask = torch.full((len(phrases),), self.task_id, dtype=torch.int32).to(self.device)
            with torch.no_grad():
                model_output = self.model(**encoded_input, adapter_mask=adapter_mask)
            embeddings = self.__mean_pooling__(model_output, encoded_input["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            tous_embeddings.append(embeddings.cpu())
        tous_embeddings = torch.cat(tous_embeddings, dim=0)
        return tous_embeddings




# def main():
    # cornichon = os.path.join(resultats_chemin, 'corpus_en_phrases.pickle')
    # indice_chemin = os.path.join(resultats_chemin, 'vecteurs.index')
    # with open(cornichon, 'rb') as f:
        # phrases = pickle.load(f)
    # sentence_embeddings = encoder_par_batch(phrases, 64)
    # créer_index_faiss(sentence_embeddings, indice_chemin)


# if __name__ == "__main__":
    # main()