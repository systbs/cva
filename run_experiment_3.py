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
NUM_EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MAX_LEN = 256

# =============================================================================
# 2. Dataset Handling (Same as Experiment 1)
# =============================================================================

def download_and_extract_imdb(url, root='.'):
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
    word_counts = Counter(word for text in texts for word in text.split())
    most_common_words = word_counts.most_common(vocab_size - 2)
    vocab = {word: i + 2 for i, (word, _) in enumerate(most_common_words)}
    vocab['<pad>'] = 0
    vocab['<unk>'] = 1
    return vocab

class IMDbDataset(Dataset):
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
        tokens = [self.vocab.get(word, self.vocab['<unk>']) for word in text.split()]
        if len(tokens) < self.max_len:
            tokens.extend([self.vocab['<pad>']] * (self.max_len - len(tokens)))
        else:
            tokens = tokens[:self.max_len]
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.float)

# =============================================================================
# 3. Model Definitions
# =============================================================================

class CompositeVectorAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.qv_proj = nn.Linear(embed_dim, embed_dim * 2)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        q, v = self.qv_proj(x).chunk(2, dim=-1)
        composite_vector = torch.mean(v, dim=1, keepdim=True)
        output = q * composite_vector
        output = self.out_proj(output)
        return output

class CVATransformerClassifier(nn.Module):
    """
    CVA Classifier with a switch for positional encoding.
    """
    def __init__(self, vocab_size, embed_dim, max_len, use_pos_encoding=True):
        super().__init__()
        self.use_pos_encoding = use_pos_encoding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        if self.use_pos_encoding:
            self.pos_encoder = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        
        self.cva_layer = CompositeVectorAttention(embed_dim)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(embed_dim, 1)
        
    def forward(self, x):
        # x shape: (B, S)
        x = self.embedding(x)  # (B, S, E)
        
        if self.use_pos_encoding:
            # Add positional encoding only if enabled
            x = x + self.pos_encoder[:, :x.size(1), :]
        
        x = self.cva_layer(x) # (B, S, E)
        x = x.mean(dim=1) # (B, E)
        x = self.dropout(x)
        return self.fc(x)

# =============================================================================
# 4. Training and Evaluation Loops (Same as Experiment 1)
# =============================================================================

def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for texts, labels in tqdm(dataloader, desc="Training"):
        texts, labels = texts.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for texts, labels in dataloader:
            texts, labels = texts.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
            outputs = model(texts)
            preds = torch.round(torch.sigmoid(outputs))
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds)
    return accuracy, f1

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

    # --- 3. Initialize Models for Comparison ---
    models_to_test = {
        "CVA with Positional Encoding": CVATransformerClassifier(VOCAB_SIZE, EMBED_DIM, MAX_LEN, use_pos_encoding=True),
        "CVA without Positional Encoding": CVATransformerClassifier(VOCAB_SIZE, EMBED_DIM, MAX_LEN, use_pos_encoding=False),
    }

    criterion = nn.BCEWithLogitsLoss()
    results = {}

    # --- 4. Run Training and Evaluation Loop ---
    for model_name, model in models_to_test.items():
        print(f"\n{'='*20} Testing Model: {model_name} {'='*20}")
        model.to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        for epoch in range(NUM_EPOCHS):
            train_loss = train_epoch(model, train_loader, optimizer, criterion)
            test_acc, test_f1 = evaluate(model, test_loader, criterion)
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")
        
        # Store final results
        final_acc, final_f1 = evaluate(model, test_loader, criterion)
        results[model_name] = {'Accuracy': final_acc, 'F1 Score': final_f1}
    
    # --- 5. Print Final Comparison Table ---
    print("\n\n" + "="*60)
    print("           ABLATION STUDY FINAL RESULTS")
    print("="*60)
    print(f"{'Model Configuration':<35} | {'Accuracy':<12} | {'F1 Score':<12}")
    print("-"*60)
    for model_name, metrics in results.items():
        print(f"{model_name:<35} | {metrics['Accuracy']:.4f}{'':<6} | {metrics['F1 Score']:.4f}")
    print("="*60)
