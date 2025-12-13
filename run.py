# ==============================================================================
# PART 3: INFERENCE (SAFE MODE - LOADS VOCAB.JSON)
# ==============================================================================
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import json
import os

# 1. SETTINGS
CHECKPOINT_PATH = "model_epoch_38.pth" 
VOCAB_PATH = "vocab.json"  # ⚠️ Make sure this file exists!
device = torch.device("cpu")
# 2. HELPER: LOAD VOCABULARY
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
        # Only needed if you want to test encoding, but inference uses decode mostly
        text = text.lower().replace(".", " .").replace(",", " ,")
        return [self.word2idx.get(w, self.word2idx.get("<UNK>", 3)) for w in text.split()]

# Initialize Tokenizer FIRST
if os.path.exists(VOCAB_PATH):
    tokenizer = JSONTokenizer(VOCAB_PATH)
else:
    raise FileNotFoundError(f"❌ Could not find {VOCAB_PATH}. Please upload it!")

# 3. REBUILD MODEL ARCHITECTURE
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

# 4. LOAD FUNCTIONS
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def load_model():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"⚠️ Error: file {CHECKPOINT_PATH} not found.")
        return None
        
    print("⏳ Loading model weights...")
    # ⚠️ CRITICAL: Use tokenizer.vocab_size from the JSON
    model = PlantGenerator(tokenizer.vocab_size).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    print(f"✅ Model Loaded Successfully!")
    return model

def predict_plant_disease(model, image_path, question):
    image = Image.open(image_path).convert("RGB")
    
    # 1. Prepare Inputs
    inputs = processor(text=[question], images=image, return_tensors="pt", padding="max_length", truncation=True)
    pix = inputs['pixel_values'].to(device)
    ids = inputs['input_ids'].to(device)
    mask = inputs['attention_mask'].to(device)
    
    # 2. Inference Loop
    curr_token = torch.tensor([[1]]).to(device) # <SOS>
    
    with torch.no_grad():
        img_f = model.clip.get_image_features(pix)
        txt_f = model.clip.get_text_features(ids, mask)
        
        context = model.reasoning(torch.cat((img_f, txt_f), dim=1))
        hidden = context.unsqueeze(0)
        
        generated_words = []
        
        for _ in range(200): 
            embed = model.embedding(curr_token)
            rnn_input = torch.cat((embed, context.unsqueeze(1)), dim=2)
            
            out, hidden = model.decoder(rnn_input, hidden)
            next_token_id = torch.argmax(model.fc_out(out), dim=2)
            
            token_id = next_token_id.item()
            if token_id == 2: break # <EOS>
            word = tokenizer.idx2word.get(token_id, "<UNK>")
            generated_words.append(word)
            curr_token = next_token_id
            
    return " ".join(generated_words)
if __name__ == "__main__":
    model = load_model()
    
    while True:
        print("\n" + "="*40)
        img_path = input("📂 Enter path to image (or 'q' to quit): ").strip().strip('"') # strips quotes if copied as path
        if img_path.lower() == 'q': break
        
        if not os.path.exists(img_path):
            print("❌ File not found, try again.")
            continue
            
        question = input("❓ Enter your question: ")
        print("🌿 Thinking...")
        
        diagnosis = predict_plant_disease(model, img_path, question)
        print(f"\n🤖 DIAGNOSIS: {diagnosis}")