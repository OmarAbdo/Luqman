import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np


# Custom Dataset class for managing stock data
class StockDataset(Dataset):
    def __init__(
        self,
        sequences,
        contexts_macro,
        contexts_fundamental,
        contexts_sentiment,
        targets,
    ):
        # Initialize with sequences, contexts, and targets
        self.sequences = sequences
        self.contexts_macro = contexts_macro
        self.contexts_fundamental = contexts_fundamental
        self.contexts_sentiment = contexts_sentiment
        self.targets = targets

    def __len__(self):
        # Returns the total number of data points
        return len(self.targets)

    def __getitem__(self, idx):
        # Retrieves the item at index 'idx'
        return (
            self.sequences[idx],
            self.contexts_macro[idx],
            self.contexts_fundamental[idx],
            self.contexts_sentiment[idx],
            self.targets[idx],
        )


# Hierarchical Attention Network class for stock prediction
class HierarchicalAttentionNetwork(nn.Module):
    def __init__(
        self,
        input_dim,
        macro_context_dim,
        fundamental_context_dim,
        sentiment_context_dim,
        embed_dim,
        n_heads,
        n_layers,
    ):
        super(HierarchicalAttentionNetwork, self).__init__()

        # Embedding layers to convert raw features to embedded representations of specified dimensions
        self.sequence_embedding = nn.Linear(
            input_dim, embed_dim
        )  # Embeds time-series data into a fixed-size representation
        self.macro_context_embedding = nn.Linear(
            macro_context_dim, embed_dim
        )  # Embeds macroeconomic data
        self.fundamental_context_embedding = nn.Linear(
            fundamental_context_dim, embed_dim
        )  # Embeds fundamental data
        self.sentiment_context_embedding = nn.Linear(
            sentiment_context_dim, embed_dim
        )  # Embeds sentiment data

        # Transformer encoder layer for processing time-series data
        # This layer helps in capturing the temporal dependencies in the sequence data
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads)
        self.sequence_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )  # Stack of Transformer encoders

        # Multihead attention mechanism to combine the encoded sequence data with the context
        # This helps in focusing on important parts of the input when combined with contextual information
        self.context_attention = nn.MultiheadAttention(embed_dim, n_heads)

        # Fully connected layer to output a single value for classification
        self.fc_out = nn.Linear(embed_dim, 1)
        # Sigmoid activation function to produce output between 0 and 1 (for binary classification)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self, sequences, macro_contexts, fundamental_contexts, sentiment_contexts
    ):
        # sequences: [batch_size, seq_len, input_dim]
        # macro_contexts: [batch_size, macro_context_dim]
        # fundamental_contexts: [batch_size, fundamental_context_dim]
        # sentiment_contexts: [batch_size, sentiment_context_dim]

        # Embedding sequences and contexts
        sequences = self.sequence_embedding(
            sequences
        )  # Transform raw sequence input to embedded form [batch_size, seq_len, embed_dim]

        # Embed macroeconomic, fundamental, and sentiment context
        macro_contexts = self.macro_context_embedding(macro_contexts).unsqueeze(
            1
        )  # Add an extra dimension for consistency [batch_size, 1, embed_dim]
        fundamental_contexts = self.fundamental_context_embedding(
            fundamental_contexts
        ).unsqueeze(
            1
        )  # [batch_size, 1, embed_dim]
        sentiment_contexts = self.sentiment_context_embedding(
            sentiment_contexts
        ).unsqueeze(
            1
        )  # [batch_size, 1, embed_dim]

        # Process sequences through Transformer encoder to capture relationships in time-series data
        sequences_encoded = self.sequence_encoder(
            sequences
        )  # [batch_size, seq_len, embed_dim]

        # Expand contexts to match sequence length
        # This allows every time-step in the sequence to have the corresponding context available
        macro_context_expanded = macro_contexts.expand(
            -1, sequences_encoded.size(1), -1
        )  # [batch_size, seq_len, embed_dim]
        fundamental_context_expanded = fundamental_contexts.expand(
            -1, sequences_encoded.size(1), -1
        )  # [batch_size, seq_len, embed_dim]
        sentiment_context_expanded = sentiment_contexts.expand(
            -1, sequences_encoded.size(1), -1
        )  # [batch_size, seq_len, embed_dim]

        # Combine all expanded contexts (adding them element-wise)
        combined_context = (
            macro_context_expanded
            + fundamental_context_expanded
            + sentiment_context_expanded
        )

        # Apply attention mechanism to combine sequence and context features
        # Attention allows the model to weigh different parts of the input differently based on context
        combined, _ = self.context_attention(
            sequences_encoded, combined_context, combined_context
        )

        # Pooling (mean) to create a fixed-size representation for the whole sequence
        # Reduces the sequence to a single representation vector by averaging over all time-steps
        combined_pooled = combined.mean(dim=1)  # [batch_size, embed_dim]

        # Pass the pooled representation through a fully connected layer to produce a single output value
        out = self.fc_out(combined_pooled)
        return self.sigmoid(
            out
        )  # Apply sigmoid to get a probability value between 0 and 1


# Example Usage
if __name__ == "__main__":
    # Load Data (Replace with your actual data loading logic)
    # Simulating data for sequences and contexts
    sequences = np.random.rand(
        100, 60, 10
    )  # Example shape: [num_samples, sequence_length, input_dim]
    macro_contexts = np.random.rand(
        100, 5
    )  # Example shape: [num_samples, macro_context_dim]
    fundamental_contexts = np.random.rand(
        100, 5
    )  # Example shape: [num_samples, fundamental_context_dim]
    sentiment_contexts = np.random.rand(
        100, 5
    )  # Example shape: [num_samples, sentiment_context_dim]
    targets = np.random.randint(0, 2, size=(100,))  # Binary targets for classification

    # Convert data to PyTorch tensors
    sequences_tensor = torch.tensor(sequences, dtype=torch.float32)
    macro_contexts_tensor = torch.tensor(macro_contexts, dtype=torch.float32)
    fundamental_contexts_tensor = torch.tensor(
        fundamental_contexts, dtype=torch.float32
    )
    sentiment_contexts_tensor = torch.tensor(sentiment_contexts, dtype=torch.float32)
    targets_tensor = torch.tensor(targets, dtype=torch.float32)

    # Create dataset and data loader for batch processing
    dataset = StockDataset(
        sequences_tensor,
        macro_contexts_tensor,
        fundamental_contexts_tensor,
        sentiment_contexts_tensor,
        targets_tensor,
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Initialize the model with appropriate dimensions and hyperparameters
    model = HierarchicalAttentionNetwork(
        input_dim=10,
        macro_context_dim=5,
        fundamental_context_dim=5,
        sentiment_context_dim=5,
        embed_dim=64,
        n_heads=4,
        n_layers=2,
    )
    # Binary Cross-Entropy loss for binary classification
    criterion = nn.BCELoss()
    # Adam optimizer for training
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    num_epochs = 10
    model.train()  # Set model to training mode
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in dataloader:
            (
                sequences_batch,
                macro_contexts_batch,
                fundamental_contexts_batch,
                sentiment_contexts_batch,
                targets_batch,
            ) = batch
            optimizer.zero_grad()  # Reset gradients from previous iteration
            # Forward pass through the model
            outputs = model(
                sequences_batch,
                macro_contexts_batch,
                fundamental_contexts_batch,
                sentiment_contexts_batch,
            ).squeeze()
            loss = criterion(outputs, targets_batch)  # Calculate loss
            loss.backward()  # Backpropagation to compute gradients
            optimizer.step()  # Update model parameters
            epoch_loss += loss.item()  # Accumulate loss for monitoring
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader):.4f}"
        )  # Print average loss for the epoch

    print("Training complete.")
