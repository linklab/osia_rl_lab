# Reference: https://github.com/guacomolia/ptr_net
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["PointerNetwork"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PointerNetwork(nn.Module):
    _last_embed: torch.Tensor  # (batch_size, seq_len, embd_size)
    _last_encoder_out: torch.Tensor  # (seq_len, batch_size, hidden_size)
    _last_decoder_hidden: torch.Tensor  # (batch_size, hidden_size)
    _last_decoder_hidden2: torch.Tensor | None  # (batch_size, hidden_size) | None

    def __init__(
        self,
        embed_input_size: int,
        embed_size: int,
        weight_size: int,
        answer_seq_len: int,
        hidden_size: int = 512,
        is_single_value_data: bool = True,
        is_GRU: bool = True,
    ):
        super().__init__()

        self.answer_seq_len = answer_seq_len
        self.weight_size = weight_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.is_GRU = is_GRU

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

    def setup(
        self,
        obs: torch.Tensor,
    ) -> None:
        # Convert numpy array to torch tensor
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=DEVICE)

        # (seq_len, n_features) -> (batch_size, seq_len, n_features)
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)

        # 1. Embedding
        # 1.1 Set last embed
        # obs: (batch_size, seq_len, n_features)
        self._last_embed = self.embed(obs)
        # self._last_embed: (batch_size, seq_len, embd_size)

        # 2. Encoding
        # self._last_embed: (batch_size, seq_len, embd_size)
        encoder_out, encoder_hidden = self.enc(self._last_embed)
        # encoder_out: (batch_size, seq_len, hiddne_size)
        # encoder_hidden: (num_layers * num_directions, batch_size, hidden_size)

        # 2.1 Set last encoder out
        self._last_encoder_out = encoder_out.transpose(1, 0)
        # self._last_encoder_out: (seq_len, batch_size, hidden_size)

        # 2.2 Set last decoder hidden
        self._last_decoder_hidden = encoder_hidden[-1]  # decoder_hidden: (batch_size, hidden_size)
        decoder_hidden2 = None
        if not self.is_GRU:
            decoder_hidden2 = to_var(torch.zeros_like(self._last_decoder_hidden))  # decoder_hidden2: (batch_size, hidden_size)
        self._last_decoder_hidden2 = decoder_hidden2

    def get_action(
        self,
        action: int | np.ndarray | None,
        action_mask: np.ndarray,
        epsilon: float,
    ) -> np.ndarray:

        # 1. Decoding
        # 1.1 Set decoder input
        if action is None:
            decoder_input = to_var(torch.zeros_like(self._last_embed[:, 0, :]))
        else:
            if isinstance(action, int):
                action = np.array([action])
            indices = to_var(torch.tensor(action).view(-1, 1, 1))  # (batch_size, 1, 1)
            indices = indices.expand(size=(-1, -1, self.embed_size))  # (batch_size, 1, embd_size)
            decoder_input = self._last_embed.gather(dim=1, index=indices)  # (batch_size, 1, embd_size)
            decoder_input = decoder_input.squeeze(dim=1)  # (batch_size, embd_size)

        # 1.2 Decoding
        if self.is_GRU:
            self._last_decoder_hidden = self.dec(
                decoder_input, self._last_decoder_hidden
            )
        else:
            self._last_decoder_hidden, self._last_decoder_hidden2 = self.dec(
                decoder_input, (self._last_decoder_hidden, self._last_decoder_hidden2)
            )

        # 2. Get random action
        if np.random.random() < epsilon:
            if action_mask.ndim == 1:
                # Add batch dimension
                action_mask = np.expand_dims(action_mask, axis=0)
            random_scores = np.random.rand(*action_mask.shape)
            random_scores[action_mask == 1] = -np.inf
            actions = np.argmax(random_scores, axis=-1)
            if actions.size == 1:
                actions = actions.item()
            return actions

        # 2. Compute blended representation at each decoder time step
        # self._last_encoder_out: (seq_len, batch_size, hidden_size)
        blend1 = self.W1(self._last_encoder_out)
        blend2 = self.W2(self._last_decoder_hidden)
        # blend1: (seq_len, batch_size, W)
        # blend2: (batch_size, W)
        blend_sum = F.tanh(blend1 + blend2)
        # blend_sum: (seq_len, batch_size, W)

        # self.vt(blend_sum): (seq_len, batch_size, 1)
        # self.vt(blend_sum).squeeze(2): (seq_len, batch_size)
        probs = self.vt(blend_sum).squeeze(2)
        probs = probs.transpose(0, 1).contiguous()  # probs: (batch_size, seq_len)

        # 3. Get action
        # 3.1 Set mask
        if action_mask.ndim == 1:
            # Add batch dimension
            # action_mask = np.expand_dims(action_mask, axis=0)
            action_mask = action_mask[np.newaxis, :]
        mask = torch.tensor(action_mask, dtype=torch.float32, device=DEVICE)  # mask: (batch_size, seq_len)
        # 3.2 Mask probs
        probs = probs.masked_fill(mask == 1, -1e6)  # probs: (batch_size, seq_len)
        # probs = F.log_softmax(out, dim=-1)  # (batch_size, seq_len)

        # 3.3 Get action
        actions = torch.argmax(probs, dim=-1).cpu().numpy()  # len(actions) = batch_size

        if actions.size == 1:
            actions = actions.item()

        return actions

    def _forward(self, input_: torch.Tensor) -> torch.Tensor:
        # Convert numpy array to torch tensor
        if isinstance(input_, np.ndarray):
            input_ = torch.tensor(input_, dtype=torch.float32, device=DEVICE)

        # (seq_len, n_features) -> (batch_size, seq_len, n_features)
        if input_.dim() == 2:
            input_ = input_.unsqueeze(0)

        batch_size = input_.size(0)
        input_seq_len = input_.size(1)

        # 1. Embedding
        # input_: (batch_size, seq_len, n_features)
        embed = self.embed(input_)
        # embed: (batch_size, seq_len, embd_size)

        # 2. Encoding
        # embed: (batch_size, seq_len, embd_size)
        decoder_input, encoder_out, encoder_hidden = self._process_encoder_step(embed)
        # decoder_input: (batch_size, embd_size)
        # encoder_out: (seq_len, batch_size, hidden_size)
        # encoder_hidden: (num_layers * num_directions, batch_size, hidden_size)

        # 2.1 Initialize mask
        mask = torch.zeros([batch_size, input_seq_len]).detach()  # mask: (batch_size, seq_len)

        # 2.2 Initialize hidden state of decoder
        decoder_hidden = encoder_hidden[-1]  # decoder_hidden: (batch_size, hidden_size)
        decoder_hidden2 = None
        if not self.is_GRU:
            decoder_hidden2 = to_var(torch.zeros(batch_size, self.hidden_size))  # decoder_hidden2: (batch_size, hidden_size)

        probs_list = []
        for _ in range(self.answer_seq_len):
            # 3. Decoding
            decoder_input, decoder_hidden, decoder_hidden2, probs = self._process_decoder_step(
                decoder_input=decoder_input,  # (batch_size, embd_size)
                decoder_hidden=decoder_hidden,  # (batch_size, hidden_size)
                decoder_hidden2=decoder_hidden2,  # (batch_size, hidden_size)
                encoder_out=encoder_out,  # (seq_len, batch_size, hidden_size)
                mask=mask,  # (batch_size, seq_len)
                embed=embed,  # (batch_size, seq_len, embd_size)
            )
            probs_list.append(probs)

        # probs: seq_len * out size = seq_len * (batch_size, seq_len) = (batch_size, seq_len, seq_len)
        probs_list = torch.stack(probs_list, dim=1)  # (batch_size, seq_len, seq_len)

        return probs_list

    def _process_encoder_step(
        self,
        embed: torch.Tensor,  # (batch_size, seq_len, embd_size)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # embed: (batch_size, seq_len, embd_size)
        encoder_out, encoder_hidden = self.enc(embed)
        # encoder_out: (batch_size, seq_len, hiddne_size)
        # encoder_hidden: (num_layers * num_directions, batch_size, hidden_size)
        encoder_out = encoder_out.transpose(1, 0)
        # encoder_out: (seq_len, batch_size, hidden_size)

        # embed: (batch_size, seq_len, embd_size)
        decoder_input = to_var(embed[:, 0, :])
        # decoder_input: (batch_size, embd_size)

        return (
            decoder_input,  # (batch_size, embd_size)
            encoder_out,  # (seq_len, batch_size, hidden_size)
            encoder_hidden,  # (num_layers * num_directions, batch_size, hidden_size)
        )

    def _process_decoder_step(
        self,
        decoder_input: torch.Tensor,  # (batch_size, embd_size)
        decoder_hidden: torch.Tensor,  # (batch_size, hidden_size)
        decoder_hidden2: torch.Tensor | None,  # (batch_size, hidden_size)
        encoder_out: torch.Tensor,  # (seq_len, batch_size, hidden_size)
        mask: torch.Tensor,  # (batch_size, seq_len)
        embed: torch.Tensor,  # (batch_size, seq_len, embd_size)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

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

        # indices.shape: (250,)
        _, indices = torch.max(probs, dim=-1)  # len(indices) = batch_size
        # mask: (batch_size, seq_len)
        mask = mask.scatter(dim=-1, index=indices.unsqueeze(-1), value=1)

        indices = indices.view(-1, 1, 1)  # (batch_size, 1, 1)
        indices = indices.expand(size=(-1, -1, self.embed_size))  # (batch_size, 1, embd_size)

        decoder_input = embed.gather(dim=1, index=indices).squeeze(dim=1)  # (batch_size, embd_size)

        return (
            decoder_input,  # (batch_size, embd_size)
            decoder_hidden,  # (batch_size, hidden_size)
            decoder_hidden2,  # (batch_size, hidden_size) | None
            probs,  # (batch_size, seq_len)
        )


def to_var(x: torch.Tensor) -> torch.Tensor:
    if torch.cuda.is_available():
        x = x.cuda()
    return x
