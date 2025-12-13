import torch
import torch.nn as nn
from transformers import CLIPModel
import json

class JSONTokenizer:
    def __init__(self, json_path):
        print(f"📖 Loading vocabulary from {json_path}...")
        with open(json_path, 'r') as f:
            self.word2idx = json.load(f)
        # Flip the dictionary to get ID -> Word
        self.idx2word = {int(v): k for k, v in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        print(f"✅ Vocab Loaded. Size: {self.vocab_size}")

    def encode(self, text):
        text = text.lower().replace(".", " .").replace(",", " ,")
        return [self.word2idx.get(w, self.word2idx.get("<UNK>", 3)) for w in text.split()]

class RecursiveReasoningLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_steps=3):
        super().__init__()
        self.num_steps = num_steps
        self.gru = nn.GRUCell(input_dim, hidden_dim)
    def forward(self, x):
        hidden = torch.zeros(x.size(0), self.gru.hidden_size).to(x.device)
        for _ in range(self.num_steps): hidden = self.gru(x, hidden)
        return hidden

class PlantGenerator(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.reasoning = RecursiveReasoningLayer(1024, 512)
        self.embedding = nn.Embedding(vocab_size, 256)
        self.decoder = nn.GRU(256 + 512, 512, batch_first=True)
        self.fc_out = nn.Linear(512, vocab_size)
