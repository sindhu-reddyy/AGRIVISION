import torch
from transformers import CLIPProcessor
from PIL import Image
import os
from .model import PlantGenerator, JSONTokenizer

# Global variables to hold model and tokenizer
model = None
tokenizer = None
device = torch.device("cpu") # Default to CPU, can be changed
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def load_resources(checkpoint_path="model_epoch_38.pth", vocab_path="vocab.json"):
    global model, tokenizer, device
    
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"❌ Could not find {vocab_path}.")
    
    tokenizer = JSONTokenizer(vocab_path)
    
    if not os.path.exists(checkpoint_path):
        # Allow running without model for testing if needed, but warn
        print(f"⚠️ Error: file {checkpoint_path} not found. Inference will fail.")
        return
        
    print("⏳ Loading model weights...")
    model = PlantGenerator(tokenizer.vocab_size).to(device)
    # map_location='cpu' is safer if user doesn't have CUDA setup right now
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"✅ Model Loaded Successfully!")

def predict_plant_disease(image_path_or_file, question):
    global model, tokenizer, device, processor
    
    if model is None:
        raise RuntimeError("Model not loaded.")
        
    try:
        image = Image.open(image_path_or_file).convert("RGB")
    except Exception as e:
        return f"Error loading image: {str(e)}"
    
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
            
    result = " ".join(generated_words)
    print(f"DEBUG: Generated {len(generated_words)} tokens.")
    print(f"DEBUG: Words: {generated_words}")
    
    if not result or len(result.strip()) == 0:
        print("⚠️ WARNING: Empty prediction generated!")
        return "I could not definitively identify the issue. Please try another image."

    print(f"✅ Prediction generated: {result}")
    return result
