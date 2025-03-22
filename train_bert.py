import json
import torch
import time
import signal
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from datetime import datetime, timedelta

# Global flag for handling interruption
running = True

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

def signal_handler(signum, frame):
    global running
    running = False
    print("\nReceived interrupt signal. Will save and exit after current batch...")

def train():
    global running
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    print("\n=== Starting Fresh BERT Training for FINRA Compliance ===")
    print("Use Ctrl+C to save and exit")
    
    # Clear CUDA cache if using GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Load data
    print("\nLoading training data...")
    with open('fcd.json', 'r') as f:
        data = json.load(f)
    
    texts = []
    labels = []
    for item in data:
        texts.extend([item['non_compliant'], item['compliant']])
        labels.extend([1, 0])
    
    total_examples = len(texts)
    print(f"\nDataset size: {total_examples} examples")
    
    # Initialize fresh model and tokenizer
    print("\nInitializing new BERT model and tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', 
                                                         num_labels=2,
                                                         output_attentions=False,
                                                         output_hidden_states=False)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print("\nTokenizing texts (this might take a few minutes)...")
    encodings = tokenizer(texts, 
                         truncation=True, 
                         padding=True, 
                         max_length=512,
                         return_tensors='pt')
    
    dataset = TensorDataset(
        encodings['input_ids'],
        encodings['attention_mask'],
        torch.tensor(labels)
    )
    
    # Training parameters
    batch_size = 16
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    epochs = 3
    
    # Rest of your training code remains the same...
    # [Previous training loop code]

    try:
        for epoch in range(epochs):
            if not running:
                break
                
            epoch_start = time.time()
            total_loss = 0
            batch_count = 0
            
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("=" * 50)
            
            for batch in dataloader:
                if not running:
                    break
                    
                batch_count += 1
                
                # Move batch to GPU if available
                batch = tuple(t.to(device) for t in batch)
                
                optimizer.zero_grad()
                outputs = model(
                    input_ids=batch[0],
                    attention_mask=batch[1],
                    labels=batch[2]
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                loss.backward()
                optimizer.step()
                
                # Progress update
                if batch_count % 10 == 0:
                    avg_loss = total_loss / batch_count
                    print(f"\rBatch {batch_count}/{len(dataloader)} | Loss: {avg_loss:.4f}", end="")
            
            print(f"\nEpoch {epoch+1} completed | Average loss: {total_loss/len(dataloader):.4f}")
    
    finally:
        print("\nSaving new model and tokenizer...")
        torch.save(model.state_dict(), 'finra_compliance_model_new.pth')
        tokenizer.save_pretrained('finra_tokenizer_new')
        print("New model saved successfully!")

if __name__ == "__main__":
    train()
