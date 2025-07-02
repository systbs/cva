import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# =============================================================================
# 1. Global Settings & Hyperparameters
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Hyperparameters for the synthetic task
VOCAB_SIZE = 100  # e.g., 50 keys and 50 values
NUM_PAIRS = 5     # Number of key-value pairs in the content
SEQUENCE_LENGTH = NUM_PAIRS * 2 + 2  # (k,v)*n + separator + query
EMBED_DIM = 64
NUM_HEADS = 4
HIDDEN_DIM = 128
LEARNING_RATE = 0.001
BATCH_SIZE = 128
TRAINING_STEPS = 5000
EVAL_STEPS = 500

# Special tokens
PAD_TOKEN = 0
SEPARATOR_TOKEN = 1
# Vocab distribution:
# 0: PAD, 1: SEP
# 2 to 51: Keys (50 total)
# 52 to 101: Values (50 total)
KEY_START_IDX = 2
VALUE_START_IDX = 52

# =============================================================================
# 2. Data Generation
# =============================================================================

def generate_copy_task_batch(batch_size, num_pairs, vocab_size):
    """Generates a batch of data for the selective copying task."""
    keys = np.random.randint(KEY_START_IDX, VALUE_START_IDX, size=(batch_size, num_pairs))
    values = np.random.randint(VALUE_START_IDX, vocab_size, size=(batch_size, num_pairs))
    
    # The query key is one of the keys from the content
    query_indices = np.random.randint(0, num_pairs, size=(batch_size))
    query_keys = keys[np.arange(batch_size), query_indices]
    
    # The target is the value corresponding to the query key
    target_values = values[np.arange(batch_size), query_indices]
    
    # Create the input sequence: k1, v1, k2, v2, ..., SEP, QK
    input_seq = np.full((batch_size, SEQUENCE_LENGTH), PAD_TOKEN)
    for i in range(num_pairs):
        input_seq[:, i*2] = keys[:, i]
        input_seq[:, i*2+1] = values[:, i]
        
    input_seq[:, -2] = SEPARATOR_TOKEN
    input_seq[:, -1] = query_keys
    
    return (
        torch.tensor(input_seq, dtype=torch.long, device=DEVICE),
        torch.tensor(target_values, dtype=torch.long, device=DEVICE)
    )

# =============================================================================
# 3. Model Definitions
# =============================================================================

class CompositeVectorAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.qv_proj = nn.Linear(embed_dim, embed_dim * 2, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        q, v = self.qv_proj(x).chunk(2, dim=-1)
        context = torch.mean(v, dim=1, keepdim=True)
        output = self.out_proj(q * context)
        return output

class CVACopyingModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_TOKEN)
        self.cva_layer = CompositeVectorAttention(embed_dim)
        # The head predicts one token from the whole vocabulary
        self.fc_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.cva_layer(x)
        # Use the output embedding of the last token (the query key) for prediction
        last_token_output = x[:, -1, :]
        return self.fc_head(last_token_output)

class StandardTransformerCopyingModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_TOKEN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        # Create a padding mask for the transformer
        padding_mask = (x == PAD_TOKEN)
        x = self.embedding(x)
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        last_token_output = x[:, -1, :]
        return self.fc_head(last_token_output)

# =============================================================================
# 4. Training and Evaluation
# =============================================================================

def train_model(model, steps):
    """Trains a given model on the synthetic task."""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    for step in tqdm(range(steps), desc=f"Training {model.__class__.__name__}"):
        inputs, targets = generate_copy_task_batch(BATCH_SIZE, NUM_PAIRS, VOCAB_SIZE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

def evaluate_model(model, steps):
    """Evaluates the model's exact match accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in tqdm(range(steps), desc=f"Evaluating {model.__class__.__name__}"):
            inputs, targets = generate_copy_task_batch(BATCH_SIZE, NUM_PAIRS, VOCAB_SIZE)
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)
    return correct / total

# =============================================================================
# 5. Main Execution Block
# =============================================================================

if __name__ == '__main__':
    # Initialize models
    cva_model = CVACopyingModel(VOCAB_SIZE, EMBED_DIM).to(DEVICE)
    std_transformer_model = StandardTransformerCopyingModel(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, HIDDEN_DIM).to(DEVICE)

    models_to_test = {
        "CVA Model": cva_model,
        "Standard Transformer": std_transformer_model
    }

    results = {}

    for name, model in models_to_test.items():
        print(f"\n--- Running experiment for: {name} ---")
        train_model(model, TRAINING_STEPS)
        accuracy = evaluate_model(model, EVAL_STEPS)
        results[name] = accuracy

    # --- Print Final Comparison Table ---
    print("\n\n" + "="*60)
    print("           SELECTIVE COPYING TASK - FINAL RESULTS")
    print("="*60)
    print(f"{'Model':<25} | {'Exact Match Accuracy'}")
    print("-"*60)
    for model_name, acc in results.items():
        print(f"{model_name:<25} | {acc:.4f}")
    print("="*60)
