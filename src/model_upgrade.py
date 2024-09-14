import torch
from torch import nn

class TorchLSTM(nn.Module):
    """
     Standard LSTM implementation using PyTorch's nn.LSTM for sequence labeling tasks like NER.

     Args:
         vocab_size (int): Size of the vocabulary.
         input_size (int): Size of input embeddings.
         hidden_size (int): Size of the hidden layer.
         num_layers (int): Number of LSTM layers.
         bidirectional (bool): If True, use a bidirectional LSTM.
         n_classes (int): Number of output classes (NER labels).
     """
    def __init__(self, vocab_size: int, input_size: int = 128, hidden_size: int = 128, num_layers: int = 1, bidirectional: bool = False, n_classes: int = 9):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, input_size)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            num_layers=num_layers,
            batch_first=True)
        self.out_fc = nn.Linear(hidden_size * (bidirectional + 1), n_classes)

    def forward(self, input_ids: torch.Tensor, init_states: tuple = None) -> torch.Tensor:
        """
          Forward pass for TorchLSTM.

          Args:
              input_ids (torch.Tensor): Input tensor of token IDs.
              init_states (tuple, optional): Initial hidden and cell states.

          Returns:
              torch.Tensor: Output tensor with predictions for each token.
          """
        bs, seq_len = input_ids.shape

        x = self.embedding(input_ids)

        output, (h_states, c_states) = self.lstm(x, init_states)

        return self.out_fc(output)