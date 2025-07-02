import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# =============================================================================
# 1. Setup and Configuration
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Experiment Parameters
SEQUENCE_LENGTHS = [64, 128, 256, 512, 1024, 2048, 4096]
BATCH_SIZE = 32
EMBED_DIM = 512
NUM_HEADS = 8
HIDDEN_DIM = 2048
VOCAB_SIZE = 10000 # Dummy vocab size

# =============================================================================
# 2. Model Definitions
# =============================================================================

class CompositeVectorAttention(nn.Module):
    """The CVA Layer, as defined previously."""
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.qv_proj = nn.Linear(embed_dim, embed_dim * 2)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        q, v = self.qv_proj(x).chunk(2, dim=-1)
        if mask is not None:
            v = v.masked_fill(~mask.unsqueeze(-1), 0)
            composite_vector = v.sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1.0)
        else:
            composite_vector = torch.mean(v, dim=1, keepdim=True)
        output = q * composite_vector
        output = self.out_proj(output)
        return output

class CVATransformer(nn.Module):
    """CVA-based model flexible to sequence length."""
    def __init__(self, vocab_size, embed_dim, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        self.cva_layer = CompositeVectorAttention(embed_dim)
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x):
        padding_mask = (x != 0)
        x = self.embedding(x)
        seq_len = x.size(1)
        x = x + self.pos_encoder[:, :seq_len, :]
        x = self.cva_layer(x, mask=padding_mask)
        x = x.mean(dim=1)
        return self.fc(x)

class StandardTransformer(nn.Module):
    """Standard Transformer model flexible to sequence length."""
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x):
        src_key_padding_mask = (x == 0)
        x = self.embedding(x)
        seq_len = x.size(1)
        x = x + self.pos_encoder[:, :seq_len, :]
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = x.mean(dim=1)
        return self.fc(x)

# =============================================================================
# 3. Measurement Function
# =============================================================================

def measure_performance(model, dummy_input, dummy_labels):
    """Measures forward/backward time and peak memory usage."""
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(DEVICE)
    
    start_time = time.time()
    
    # Forward and backward pass
    output = model(dummy_input)
    loss = nn.BCEWithLogitsLoss()(output, dummy_labels)
    loss.backward()
    
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    duration_ms = (end_time - start_time) * 1000
    
    peak_memory_mb = 0
    if DEVICE.type == 'cuda':
        peak_memory_mb = torch.cuda.max_memory_allocated(DEVICE) / (1024 * 1024)
        
    return duration_ms, peak_memory_mb

# =============================================================================
# 4. Main Experiment Loop
# =============================================================================

if __name__ == '__main__':
    results = []
    max_supported_len = max(SEQUENCE_LENGTHS)

    print("Initializing models...")
    cva_model = CVATransformer(VOCAB_SIZE, EMBED_DIM, max_supported_len).to(DEVICE)
    std_model = StandardTransformer(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, HIDDEN_DIM, max_supported_len).to(DEVICE)
    
    print("Starting performance measurement...")
    for seq_len in tqdm(SEQUENCE_LENGTHS, desc="Testing Sequence Lengths"):
        # Create dummy data for the current sequence length
        dummy_input = torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, seq_len), device=DEVICE)
        dummy_labels = torch.rand((BATCH_SIZE, 1), device=DEVICE)

        # Measure CVA Transformer
        cva_model.zero_grad()
        cva_time, cva_mem = measure_performance(cva_model, dummy_input, dummy_labels)

        # Measure Standard Transformer
        std_time, std_mem = float('inf'), float('inf')
        try:
            std_model.zero_grad()
            std_time, std_mem = measure_performance(std_model, dummy_input, dummy_labels)
        except torch.cuda.OutOfMemoryError:
            print(f"Standard Transformer ran out of memory at sequence length {seq_len}.")
            # Break the loop for standard model as it will fail for longer sequences too
            for remaining_seq_len in [sl for sl in SEQUENCE_LENGTHS if sl >= seq_len]:
                 results.append({
                    'Sequence Length': remaining_seq_len,
                    'CVA Time (ms)': cva_time,
                    'CVA Memory (MB)': cva_mem,
                    'Standard Time (ms)': float('inf'),
                    'Standard Memory (MB)': float('inf'),
                })
            break # Exit the main loop
            
        results.append({
            'Sequence Length': seq_len,
            'CVA Time (ms)': cva_time,
            'CVA Memory (MB)': cva_mem,
            'Standard Time (ms)': std_time,
            'Standard Memory (MB)': std_mem,
        })
    
    # Convert results to a DataFrame for easier plotting
    results_df = pd.DataFrame(results)
    print("\n--- Performance Results ---")
    print(results_df)

    # =============================================================================
    # 5. Plotting Results
    # =============================================================================
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Processing Time
    ax1.plot(results_df['Sequence Length'], results_df['CVA Time (ms)'], 'o-', label='CVA Transformer', color='blue')
    ax1.plot(results_df['Sequence Length'], results_df['Standard Time (ms)'], 's-', label='Standard Transformer', color='red')
    ax1.set_xlabel('Sequence Length', fontsize=12)
    ax1.set_ylabel('Processing Time (ms)', fontsize=12)
    ax1.set_title('Time Complexity: Time vs. Sequence Length', fontsize=14)
    ax1.legend()
    ax1.set_yscale('log') # Use log scale to see differences better

    # Plot 2: Memory Usage
    ax2.plot(results_df['Sequence Length'], results_df['CVA Memory (MB)'], 'o-', label='CVA Transformer', color='blue')
    ax2.plot(results_df['Sequence Length'], results_df['Standard Memory (MB)'], 's-', label='Standard Transformer', color='red')
    ax2.set_xlabel('Sequence Length', fontsize=12)
    ax2.set_ylabel('Peak Memory Usage (MB)', fontsize=12)
    ax2.set_title('Memory Complexity: Memory vs. Sequence Length', fontsize=14)
    ax2.legend()
    
    fig.tight_layout()
    plt.show()
