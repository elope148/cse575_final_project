"""
Convolutional LSTM cell and multi-layer ConvLSTM.

Reference: Shi et al., "Convolutional LSTM Network: A Machine Learning
Approach for Precipitation Nowcasting", NeurIPS 2015.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple


class ConvLSTMCell(nn.Module):
    """
    Single ConvLSTM cell.

    Replaces fully-connected gates with convolutional gates to preserve
    spatial structure in the hidden state.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int = 3,
        bias: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        # Single conv computes all 4 gates at once
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, input_dim, H, W)
            state: tuple of (h, c), each (B, hidden_dim, H, W)

        Returns:
            (h_next, c_next): next hidden and cell states
        """
        B, _, H, W = x.shape

        if state is None:
            h = torch.zeros(B, self.hidden_dim, H, W, device=x.device)
            c = torch.zeros(B, self.hidden_dim, H, W, device=x.device)
        else:
            h, c = state

        combined = torch.cat([x, h], dim=1)  # (B, input_dim + hidden_dim, H, W)
        gates = self.conv(combined)            # (B, 4 * hidden_dim, H, W)

        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class ConvLSTM(nn.Module):
    """
    Multi-layer ConvLSTM.

    Processes a sequence of spatial feature maps and returns the hidden
    states from the last layer at each time step.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        kernel_size: int = 3,
        num_layers: Optional[int] = None,
        dropout: float = 0.0,
        return_all_layers: bool = False,
    ):
        super().__init__()
        if num_layers is not None:
            assert num_layers == len(hidden_dims), \
                f"num_layers ({num_layers}) != len(hidden_dims) ({len(hidden_dims)})"

        self.num_layers = len(hidden_dims)
        self.hidden_dims = hidden_dims
        self.return_all_layers = return_all_layers

        cells = []
        for i in range(self.num_layers):
            cur_input = input_dim if i == 0 else hidden_dims[i - 1]
            cells.append(ConvLSTMCell(cur_input, hidden_dims[i], kernel_size))

        self.cells = nn.ModuleList(cells)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

    def forward(
        self,
        x: torch.Tensor,
        hidden_states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: (B, T, C, H, W) input sequence
            hidden_states: list of (h, c) tuples for each layer

        Returns:
            output: (B, T, hidden_dims[-1], H, W) if return_all_layers=False
                    else list of outputs for each layer
            last_states: list of (h, c) for each layer at final time step
        """
        B, T, _, H, W = x.shape

        if hidden_states is None:
            hidden_states = [None] * self.num_layers

        layer_outputs = []
        last_states = []

        cur_input = x  # (B, T, C, H, W)

        for layer_idx, cell in enumerate(self.cells):
            h, c = hidden_states[layer_idx] if hidden_states[layer_idx] is not None else (None, None)
            state = (h, c) if h is not None else None

            outputs_t = []
            for t in range(T):
                h_next, c_next = cell(cur_input[:, t], state)
                state = (h_next, c_next)
                outputs_t.append(h_next)

            # Stack time outputs: (B, T, hidden_dim, H, W)
            layer_out = torch.stack(outputs_t, dim=1)

            if self.dropout is not None and layer_idx < self.num_layers - 1:
                # Apply dropout between layers (not on last layer)
                B2, T2, C2, H2, W2 = layer_out.shape
                layer_out = self.dropout(
                    layer_out.reshape(B2 * T2, C2, H2, W2)
                ).reshape(B2, T2, C2, H2, W2)

            cur_input = layer_out
            layer_outputs.append(layer_out)
            last_states.append(state)

        if self.return_all_layers:
            return layer_outputs, last_states
        else:
            return layer_outputs[-1], last_states
