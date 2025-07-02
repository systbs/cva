import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import Counter
from tqdm import tqdm
import numpy as np
import os
import requests
import tarfile
from sklearn.metrics import f1_score

# =============================================================================
# 1. Global Settings & Hyperparameters
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Hyperparameters
VOCAB_SIZE = 10000
EMBED_DIM = 128
HIDDEN_DIM = 256
NUM_HEADS = 4  # For standard Transformer
NUM_EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MAX_LEN = 256 # Max sequence length

# =============================================================================
# 2. Dataset Handling (without torchtext)
# =============================================================================

def download_and_extract_imdb(url, root='.'):
    """Downloads and extracts the IMDb dataset."""
    dataset_path = os.path.join(root, 'aclImdb')
    if os.path.exists(dataset_path):
        print("IMDb dataset already exists.")
        return dataset_path
    
    filename = 'aclImdb_v1.tar.gz'
    filepath = os.path.join(root, filename)
    
    print("Downloading IMDb dataset...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(filepath, 'wb') as f:
            for chunk in tqdm(r.iter_content(chunk_size=8192)):
                f.write(chunk)
    
    print("Extracting dataset...")
    with tarfile.open(filepath, 'r:gz') as tar:
        tar.extractall(path=root)
    
    os.remove(filepath)
    print("Dataset ready.")
    return dataset_path

def read_imdb_data(data_dir):
    """Reads texts and labels from the IMDb directory structure."""
    texts, labels = [], []
    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(data_dir, label_type)
        for fname in tqdm(os.listdir(dir_name), desc=f"Reading {label_type} files"):
            if fname.endswith('.txt'):
                with open(os.path.join(dir_name, fname), 'r', encoding='utf-8') as f:
                    texts.append(f.read())
                labels.append(0 if label_type == 'neg' else 1)
    return texts, labels

def build_vocab(texts, vocab_size):
    """Builds a vocabulary from the texts."""
    word_counts = Counter(word for text in texts for word in text.split())
    # Most common words, reserving 0 for padding and 1 for unknown
    most_common_words = word_counts.most_common(vocab_size - 2)
    vocab = {word: i + 2 for i, (word, _) in enumerate(most_common_words)}
    vocab['<pad>'] = 0
    vocab['<unk>'] = 1
    return vocab

class IMDbDataset(Dataset):
    """Custom Dataset class for IMDb."""
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        # Tokenize and numericalize
        tokens = [self.vocab.get(word, self.vocab['<unk>']) for word in text.split()]
        # Pad or truncate
        if len(tokens) < self.max_len:
            tokens.extend([self.vocab['<pad>']] * (self.max_len - len(tokens)))
        else:
            tokens = tokens[:self.max_len]
        
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.float)

# =============================================================================
# 3. Model Definitions
# =============================================================================

# 3.1 --- CVA Model (Your Proposed Model)
class CompositeVectorAttention(nn.Module):
    """
    Composite Vector Attention (CVA) Layer.
    Source: Based on the user-provided document.
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        # A single linear layer to project input to Q and V simultaneously
        self.qv_proj = nn.Linear(embed_dim, embed_dim * 2)
        # Output projection layer
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        # x shape: (B, S, E) -> Batch, Sequence, Embedding
        B, S, E = x.shape
        
        # Step 1: Create Q and V projections
        q, v = self.qv_proj(x).chunk(2, dim=-1)
        
        # Step 2: Create the composite vector by averaging V
        # We need to handle padding masks if provided
        if mask is not None:
            # Expand mask for broadcasting: (B, S) -> (B, S, 1)
            mask = mask.unsqueeze(-1).float()
            v = v * mask # zero out padded values
            # Calculate mean only over non-padded elements
            composite_vector = torch.sum(v, dim=1, keepdim=True) / torch.sum(mask, dim=1, keepdim=True).clamp(min=1.0)
        else:
            composite_vector = torch.mean(v, dim=1, keepdim=True)
            
        # Step 3: Gating attention operation
        output = q * composite_vector
        
        # Step 4: Final output projection
        output = self.out_proj(output)
        
        return output

class CVATransformerClassifier(nn.Module):
    """A classifier using the CVA layer."""
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoder = nn.Parameter(torch.zeros(1, MAX_LEN, embed_dim)) # Simple positional encoding
        self.cva_layer = CompositeVectorAttention(embed_dim)
        self.dropout = nn.Dropout(0.1)
        # Classifier head
        self.fc = nn.Linear(embed_dim, 1)
        
    def forward(self, x):
        # x shape: (B, S)
        padding_mask = (x != 0) # True for non-padded tokens
        x = self.embedding(x)  # (B, S, E)
        x = x + self.pos_encoder[:, :x.size(1), :] # Add positional encoding
        
        x = self.cva_layer(x, mask=padding_mask) # (B, S, E)
        
        # Use the mean of the sequence output for classification
        x = x.mean(dim=1) # (B, E)
        x = self.dropout(x)
        
        return self.fc(x)

# 3.2 --- Standard Transformer Model (For Comparison)
class StandardTransformerClassifier(nn.Module):
    """A classifier using a standard nn.TransformerEncoderLayer."""
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoder = nn.Parameter(torch.zeros(1, MAX_LEN, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim, 
            batch_first=True,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # x shape: (B, S)
        src_key_padding_mask = (x == 0) # Mask for padded tokens
        x = self.embedding(x) # (B, S, E)
        x = x + self.pos_encoder[:, :x.size(1), :]
        
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask) # (B, S, E)
        
        # Use the mean of the sequence output
        x = x.mean(dim=1) # (B, E)
        x = self.dropout(x)
        
        return self.fc(x)

# 3.3 --- LSTM Baseline Model
class LSTMClassifier(nn.Module):
    """A baseline LSTM classifier."""
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim * 2, 1) # *2 for bidirectional

    def forward(self, x):
        # x shape: (B, S)
        x = self.embedding(x) # (B, S, E)
        
        # LSTM returns output, (hidden, cell)
        _, (hidden, _) = self.lstm(x)
        
        # Concatenate final forward and backward hidden states
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = self.dropout(hidden)
        
        return self.fc(hidden)

# =============================================================================
# 4. Training and Evaluation Loops
# =============================================================================

def train_epoch(model, dataloader, optimizer, criterion):
    """Performs one training epoch."""
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for texts, labels in tqdm(dataloader, desc="Training"):
        texts, labels = texts.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.round(torch.sigmoid(outputs))
        all_preds.extend(preds.cpu().detach().numpy())
        all_labels.extend(labels.cpu().detach().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    return avg_loss, accuracy

def evaluate(model, dataloader, criterion):
    """Evaluates the model on the given dataset."""
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for texts, labels in tqdm(dataloader, desc="Evaluating"):
            texts, labels = texts.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
            
            outputs = model(texts)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            preds = torch.round(torch.sigmoid(outputs))
            all_preds.extend(preds.cpu().detach().numpy())
            all_labels.extend(labels.cpu().detach().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds)
    return avg_loss, accuracy, f1

# =============================================================================
# 5. Main Execution Block
# =============================================================================

if __name__ == '__main__':
    # --- 1. Load Data ---
    dataset_url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    dataset_path = download_and_extract_imdb(dataset_url)
    
    train_texts, train_labels = read_imdb_data(os.path.join(dataset_path, 'train'))
    test_texts, test_labels = read_imdb_data(os.path.join(dataset_path, 'test'))

    # --- 2. Build Vocab and Datasets ---
    vocab = build_vocab(train_texts, VOCAB_SIZE)
    train_dataset = IMDbDataset(train_texts, train_labels, vocab, MAX_LEN)
    test_dataset = IMDbDataset(test_texts, test_labels, vocab, MAX_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # --- 3. Initialize Models, Optimizer, Criterion ---
    models_to_test = {
        "LSTM Baseline": LSTMClassifier(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM),
        "Standard Transformer": StandardTransformerClassifier(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, HIDDEN_DIM),
        "CVA Transformer": CVATransformerClassifier(VOCAB_SIZE, EMBED_DIM),
    }

    criterion = nn.BCEWithLogitsLoss()
    results = {}

    # --- 4. Run Training and Evaluation Loop for each model ---
    for model_name, model in models_to_test.items():
        print(f"\n{'='*20} Testing Model: {model_name} {'='*20}")
        model.to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        best_f1 = 0
        for epoch in range(NUM_EPOCHS):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
            test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion)
            
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")
            
            if test_f1 > best_f1:
                best_f1 = test_f1
                results[model_name] = {'Accuracy': test_acc, 'F1 Score': test_f1}
    
    # --- 5. Print Final Results ---
    print("\n\n" + "="*50)
    print("           FINAL RESULTS COMPARISON")
    print("="*50)
    print(f"{'Model':<25} | {'Accuracy':<12} | {'F1 Score':<12}")
    print("-"*50)
    for model_name, metrics in results.items():
        print(f"{model_name:<25} | {metrics['Accuracy']:.4f}{'':<6} | {metrics['F1 Score']:.4f}")
    print("="*50)

