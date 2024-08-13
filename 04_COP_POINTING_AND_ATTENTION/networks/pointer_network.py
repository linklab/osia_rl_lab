# Reference: https://github.com/guacomolia/ptr_net
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["PointerNetwork"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PointerNetwork(nn.Module):
    _last_decoder_input: torch.Tensor
    _last_hidden: torch.Tensor
    _last_cell_state: torch.Tensor
    _last_encoder_states: torch.Tensor

    def __init__(
        self,
        embed_input_size: int,
        embed_size: int,
        weight_size: int,
        answer_seq_len: int,
        hidden_size: int = 512,
        is_single_value_data: bool = True,
        is_GRU: bool = True,
        decoder_input_always_zero: bool = True,
    ):
        super().__init__()

        self.answer_seq_len = answer_seq_len
        self.weight_size = weight_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.is_GRU = is_GRU
        self.decoder_input_always_zero = decoder_input_always_zero

        if is_single_value_data:
            self.embed = nn.Embedding(embed_input_size, embed_size)
        else:
            self.embed = nn.Linear(embed_input_size, embed_size)

        if is_GRU:
            # GRUCell's input is always batch first
            self.enc = nn.GRU(input_size=embed_size, hidden_size=hidden_size, batch_first=True)   # num_layers=1
            self.dec = nn.GRUCell(input_size=embed_size, hidden_size=hidden_size)
        else:
            # LSTMCell's input is always batch first
            self.enc = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, batch_first=True)  # num_layers=1
            self.dec = nn.LSTMCell(input_size=embed_size, hidden_size=hidden_size)

        self.W1 = nn.Linear(in_features=hidden_size, out_features=weight_size, bias=False)  # blending encoder
        self.W2 = nn.Linear(hidden_size, weight_size, bias=False)  # blending decoder
        self.vt = nn.Linear(weight_size, 1, bias=False)  # scaling sum of enc and dec by v.T

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        # Convert numpy array to torch tensor
        if isinstance(input_, np.ndarray):
            input_ = torch.tensor(input_, dtype=torch.float32, device=DEVICE)

        # (seq_len, n_features) -> (batch_size, seq_len, n_features)
        if input_.dim() == 2:
            input_ = input_.unsqueeze(0)

        batch_size = input_.size(0)
        input_seq_len = input_.size(1)

        # 1. Embedding
        # (batch_dize, seq_len, n_features) * (n_features, embd_size) = (batch_size, seq_len, embd_size)
        embed = self.embed(input_)

        # 2. Encoding
        decoder_input, encoder_out, encoder_hidden = self.process_encoder_step(embed)

        # 2.1 Initialize mask
        mask = torch.zeros([batch_size, input_seq_len]).detach()  # (batch_size, seq_len)

        # 2.2 Initialize hidden state of decoder
        decoder_hidden = encoder_hidden[-1]  # (batch_size, hidden_size)
        decoder_hidden2 = None
        if not self.is_GRU:
            decoder_hidden2 = to_var(torch.zeros(batch_size, self.hidden_size))

        probs_list = []
        for _ in range(self.answer_seq_len):
            # 3. Decoding
            decoder_input, decoder_hidden, decoder_hidden2, probs = self.process_decoder_step(
                decoder_input=decoder_input,
                decoder_hidden=decoder_hidden,
                decoder_hidden2=decoder_hidden2,
                encoder_out=encoder_out,
                mask=mask,
                embed=embed,
            )
            probs_list.append(probs)

        # probs: seq_len * out size = seq_len * (batch_size, seq_len) = (batch_size, seq_len, seq_len)
        probs_list = torch.stack(probs_list, dim=1)  # (batch_size, seq_len, seq_len)

        return probs_list

    def process_encoder_step(
        self,
        embed: torch.Tensor,  # (batch_size, seq_len, embd_size)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = embed.size(0)
        # encoder_out: (batch_size, seq_len, hiddne_size)
        # hc: (num_layers * num_directions, batch_size, hidden_size)
        encoder_out, encoder_hidden = self.enc(embed)
        encoder_out = encoder_out.transpose(1, 0)  # (seq_len, batch_size, hidden_size)

        if self.decoder_input_always_zero:
            decoder_input = to_var(torch.zeros(batch_size, self.embed_size))  # (batch_size, embd_size)
        else:
            decoder_input = to_var(embed[:, 0, :])  # (batch_size, embd_size)

        return (
            decoder_input,  # (batch_size, embd_size)
            encoder_out,  # (seq_len, batch_size, hidden_size)
            encoder_hidden,  # (num_layers * num_directions, batch_size, hidden_size)
        )

    def process_decoder_step(
        self,
        decoder_input: torch.Tensor,  # (batch_size, embd_size)
        decoder_hidden: torch.Tensor,  # (batch_size, hidden_size)
        decoder_hidden2: torch.Tensor | None,  # (batch_size, hidden_size)
        encoder_out: torch.Tensor,  # (seq_len, batch_size, hidden_size)
        mask: torch.Tensor,  # (batch_size, seq_len)
        embed: torch.Tensor,  # (batch_size, seq_len, embd_size)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = decoder_input.size(0)

        if self.is_GRU:
            # (batch_size, embd_size) * (embd_size, hidden_size) = (batch_size, hidden_size)
            decoder_hidden = self.dec(decoder_input, decoder_hidden)
        else:
            decoder_hidden, decoder_hidden2 = self.dec(decoder_input, (decoder_hidden, decoder_hidden2))
        # hidden: (batch_size, hidden_size)

        # Compute blended representation at each decoder time step
        blend1 = self.W1(encoder_out)  # (seq_len, batch_size, hidden_size) * (hidden_size, W) = (seq_len, batch_size, W)
        blend2 = self.W2(decoder_hidden)  # (batch_size, hidden_size) * (hidden_size, W) = (batch_size, W)
        blend_sum = F.tanh(blend1 + blend2)  # (batch_size, W)
        # blend_sum: (seq_len, batch_size, W)

        # self.vt(blend_sum): (seq_len, batch_size, 1)
        # self.vt(blend_sum).squeeze(2): (seq_len, batch_size)
        probs = self.vt(blend_sum).squeeze(2)
        probs = probs.transpose(0, 1).contiguous()  # (batch_size, seq_len)
        probs = probs.masked_fill(mask == 1, -1e6)  # (batch_size, seq_len)
        # probs = F.log_softmax(out, dim=-1)  # (batch_size, seq_len)

        if self.decoder_input_always_zero:
            decoder_input = to_var(torch.zeros(batch_size, self.embed_size))  # (batch_size, embd_size)
        else:
            # indices.shape: (250,)
            _, indices = torch.max(probs, dim=-1)  # len(indices) = batch_size
            # mask: (batch_size, seq_len)
            mask = mask.scatter(dim=-1, index=indices.unsqueeze(-1), value=1)

            indices = indices.view(-1, 1, 1)  # (batch_size, 1, 1)
            indices = indices.expand(size=(-1, -1, self.embed_size))  # (batch_size, 1, embd_size)

            decoder_input = embed.gather(dim=1, index=indices).squeeze(dim=1)  # (batch_size, embd_size)

        return decoder_input, decoder_hidden, decoder_hidden2, probs


def to_var(x: torch.Tensor) -> torch.Tensor:
    if torch.cuda.is_available():
        x = x.cuda()
    return x
