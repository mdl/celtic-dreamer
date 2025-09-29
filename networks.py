import torch
import torch.nn as nn
from torch.distributions import Normal, Bernoulli, Independent, OneHotCategoricalStraightThrough, Categorical
from torch.distributions.utils import probs_to_logits
from utils import sequentialModel1D

class RecurrentModel(nn.Module):
    def __init__(self, recurrentSize: int, latentSize: int, actionSize: int, config):
        super().__init__()
        self.d_model = getattr(config, "hiddenSize", 256)
        self.n_heads = getattr(config, "nHeads", 8)
        self.num_layers = getattr(config, "numLayers", 3)
        self.ff_size = getattr(config, "ffSize", 2 * self.d_model)
        self.max_len = getattr(config, "maxSeqLen", 128)
        self.act_name = getattr(config, "activation", "gelu").lower()

        self.in_proj = nn.Linear(latentSize + actionSize, self.d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_len, self.d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.ff_size,
            activation=self.act_name,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=self.num_layers, enable_nested_tensor=False)
        self.to_state = nn.Linear(self.d_model, recurrentSize)

        self.register_buffer("mem_tokens", None, persistent=False)
        self.register_buffer("causal_mask", torch.empty(0), persistent=False)

    def reset(self):
        self.mem_tokens = None

    def forward(self, recurrentState, latentState: torch.Tensor, action: torch.Tensor):
        token = torch.cat([latentState, action], dim=-1)
        token = self.in_proj(token).unsqueeze(1)

        if self.mem_tokens is None:
            mem = token
        else:
            mem = torch.cat([self.mem_tokens, token], dim=1)
            if mem.size(1) > self.max_len:
                mem = mem[:, -self.max_len:, :]
        self.mem_tokens = mem.detach()

        T = mem.size(1)
        x = mem + self.pos_embed[:, :T]

        if (self.causal_mask.numel() == 0
            or self.causal_mask.size(0) < T
            or self.causal_mask.device != x.device
            or self.causal_mask.dtype != x.dtype):
            m = torch.full((T, T), float("-inf"), device=x.device, dtype=x.dtype)
            self.causal_mask = torch.triu(m, diagonal=1)

        x = self.encoder(x, mask=self.causal_mask[:T, :T])
        h = self.to_state(x[:, -1, :])
        return h


# class RecurrentModel(nn.Module):
#     def __init__(self, recurrentSize, latentSize, actionSize, config):
#         super().__init__()
#         self.config = config
#         self.activation = nn.ReLU()  # TanH or something
#         self.actionSize: int = actionSize
#
#         # Recurrent model works with three inputs in total -> latent state, recurrent state and the action
#         # It outputs a new recurrent state
#         # Rs_1 = RecurrentModel(Rs_0, Ls_0, a_0)
#         self.linear = nn.Linear(latentSize + actionSize, self.config.hiddenSize)
#         self.recurrent = nn.GRUCell(self.config.hiddenSize, recurrentSize)  # input size, hidden size
#
#     def forward(self, recurrentState, latentState, action):
#         catted = torch.cat((latentState, action), -1)
#
#         return self.recurrent(self.activation(self.linear(catted)),
#                               recurrentState)  # input = latent + action, hidden = recurrent state
#
#     def reset(self):
#         pass

class PriorNet(nn.Module):
    def __init__(self, inputSize, latentLength, latentClasses, config):
        super().__init__()
        self.config = config
        self.latentLength = latentLength
        self.latentClasses = latentClasses
        self.latentSize = latentLength * latentClasses
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize] * self.config.numLayers, self.latentSize,
                                         self.config.activation)

    def forward(self, x):
        rawLogits = self.network(x)

        probabilities = rawLogits.view(-1, self.latentLength, self.latentClasses).softmax(
            -1)  # Convert logits to probabilities
        uniform = torch.ones_like(
            probabilities) / self.latentClasses  # Create a uniform probability distribution for each discrete latent variable

        # This line performs a uniform mixing of probabilities. It's a regularization technique. 
        # By mixing the network's predicted probabilities with a small amount of a uniform distribution, we ensure:
        # No probability ever becomes exactly zero or one. This can help prevent the model from becoming overly confident and can improve numerical stability.
        # It encourages a small amount of exploration in the latent space, ensuring that all latent classes have at least a tiny chance of being selected. 
        finalProbabilities = (1 - self.config.uniformMix) * probabilities + self.config.uniformMix * uniform

        logits = probs_to_logits(finalProbabilities)
        sample = Independent(OneHotCategoricalStraightThrough(logits=logits), 1).rsample()

        return sample.view(-1, self.latentSize), logits


class PosteriorNet(nn.Module):
    def __init__(self, inputSize, latentLength, latentClasses, config):
        super().__init__()
        self.config = config
        self.latentLength = latentLength
        self.latentClasses = latentClasses
        self.latentSize = latentLength * latentClasses
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize] * self.config.numLayers, self.latentSize,
                                         self.config.activation)

    def forward(self, x):
        rawLogits = self.network(x)

        probabilities = rawLogits.view(-1, self.latentLength, self.latentClasses).softmax(-1)
        uniform = torch.ones_like(probabilities) / self.latentClasses
        finalProbabilities = (1 - self.config.uniformMix) * probabilities + self.config.uniformMix * uniform
        logits = probs_to_logits(finalProbabilities)

        sample = Independent(OneHotCategoricalStraightThrough(logits=logits), 1).rsample()
        return sample.view(-1, self.latentSize), logits


class RewardModel(nn.Module):
    def __init__(self, inputSize, config):
        super().__init__()
        self.config = config

        # The input is the full state, the output is the mean and std (given out as log std)
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize] * self.config.numLayers, 2,
                                         self.config.activation)

    def forward(self, x):
        mean, logStd = self.network(x).chunk(2, dim=-1)
        std = torch.nn.functional.softplus(logStd) + 1e-4
        return Normal(mean.squeeze(-1), std.squeeze(-1))


class ContinueModel(nn.Module):
    def __init__(self, inputSize, config):
        super().__init__()
        self.config = config

        # Output is 1 dimensional - yes or no
        # Creates an MLP that outputs a single value (a logit)
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize] * self.config.numLayers, 1,
                                         self.config.activation)

    def forward(self, x):
        # The output of the network is a logit.
        # It returns a Bernoulli distribution parameterized by these logits.
        # A Bernoulli distribution models a binary outcome (e.g., continue or terminate).
        # The distribution is expressed as (p for x = 1 | 1 - p for x = 0; where p is a probability)
        return Bernoulli(logits=self.network(x).squeeze(-1))


class Actor(nn.Module):
    def __init__(self, inputSize, actionSize, device, config):
        super().__init__()
        self.config = config
        self.actionSize = actionSize
        self.device = device

        # Network outputs logits for each binary action
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize] * self.config.numLayers, actionSize,
                                         self.config.activation)

    def forward(self, x, training=False):
        logits = self.network(x)

        # Create a Bernoulli distribution for each action dimension
        distribution = Categorical(logits=logits)

        # Sample from each distribution
        sample = distribution.sample()

        if training:
            # Calculate log probabilities and entropy
            logprobs = distribution.log_prob(sample)
            ent = distribution.entropy()

            return sample, logprobs, ent
        else:
            return sample


class Critic(nn.Module):
    def __init__(self, inputSize, config):
        super().__init__()
        self.config = config

        # The sequentialModel1D is configured to output 2 values (mean and log_std for the value distribution)
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize] * self.config.numLayers, 2,
                                         self.config.activation)

    def forward(self, x):
        mean, logStd = self.network(x).chunk(2, dim=-1)
        std = torch.nn.functional.softplus(logStd) + 1e-4
        return Normal(mean.squeeze(-1), std.squeeze(-1))


# TODO: critic doesn't take action

# OLD ENCODER & DECODER SETUP
# class EncoderConv(nn.Module):
#     def __init__(self, inputShape, outputSize):
#         super().__init__()
#         self.outputSize = outputSize
#         activation = nn.ReLU()
#         channels, _, __ = inputShape
#         kernelSize = 4
#         stride = 2
#         depth = 32
#
#         self.convolutionalNet = nn.Sequential(
#             nn.Conv2d(channels,    1 * depth, kernelSize, stride, padding=1), activation,  # 128→64
#             nn.Conv2d(1 * depth,       2 * depth, kernelSize, stride, padding=1), activation,  # 64 →32
#             nn.Conv2d(2 * depth,       4 * depth, kernelSize, stride, padding=1), activation,  # 32 →16
#             nn.Conv2d(4 * depth,       8 * depth, kernelSize, stride, padding=1), activation,  # 16 →8
#             nn.Conv2d(8 * depth,      32 * depth, kernelSize, stride, padding=1), activation,  # 8  →4
#             nn.Flatten(),                                           # (N, 32d*4*4)
#             nn.Linear(32 * depth * 4 * 4, outputSize), activation
#         )
#
#         # region other resolutions
#         # # new proposed 64x64
#         # self.convolutionalNet = nn.Sequential(
#         #     # Input: (3, 64, 64)
#         #     nn.Conv2d(channels, 1 * depth, kernelSize, stride, padding=1),
#         #     activation,
#         #     # -> (32, 32, 32)
#         #     nn.Conv2d(1 * depth, 2 * depth, kernelSize, stride, padding=1),
#         #     activation,
#         #     # -> (64, 16, 16)
#         #     nn.Conv2d(2 * depth, 4 * depth, kernelSize, stride, padding=1),
#         #     activation,
#         #     # -> (128, 8, 8)
#         #     nn.Conv2d(4 * depth, 8 * depth, kernelSize, stride, padding=1),
#         #     activation,
#         #     # -> (256, 4, 4)
#         #     nn.Flatten(),
#         #     nn.Linear(8 * depth * 4 * 4, self.outputSize),
#         #     activation
#         # )
#
#         # old
#         # self.convolutionalNet = nn.Sequential(
#         #     nn.Conv2d(channels, depth * 1, kernelSize, stride, padding=1), activation,
#         #     nn.Conv2d(depth * 1, depth * 2, kernelSize, stride, padding=1), activation,
#         #     nn.Conv2d(depth * 2, depth * 4, kernelSize, stride, padding=1), activation,
#         #     nn.Conv2d(depth * 4, depth * 8, kernelSize, stride, padding=1), activation,
#         #     # Add an additional convolutional layer to handle the larger input dimensions
#         #     nn.Conv2d(depth * 8, depth * 16, kernelSize, stride, padding=1), activation,
#         #     nn.Flatten(),
#         #     nn.Linear(depth * 16 * (height // (stride ** 5)) * (width // (stride ** 5)), outputSize),
#         #     activation
#         # )
#         # endregion
#
#     def forward(self, x):
#         return self.convolutionalNet(x).view(-1, self.outputSize)
#
#
# class DecoderConv(nn.Module):
#     def __init__(self, inputSize, outputShape):
#         super().__init__()
#         channels, _, __ = outputShape
#         activation = nn.ReLU()
#         kernelSize = 4
#         stride = 2
#         depth = 32
#
#         # 128x128
#         self.network = nn.Sequential(
#             nn.Linear(inputSize, depth * 32 * 4 * 4),
#             nn.Unflatten(1, (depth * 32, 4, 4)),
#             nn.ConvTranspose2d(depth * 32, depth * 8, kernelSize, stride, padding=1), activation,
#             nn.ConvTranspose2d(depth * 8, depth * 4, kernelSize, stride, padding=1), activation,
#             nn.ConvTranspose2d(depth * 4, depth * 2, kernelSize, stride, padding=1), activation,
#             nn.ConvTranspose2d(depth * 2, depth * 1, kernelSize, stride, padding=1), activation,
#             nn.ConvTranspose2d(depth * 1, channels, kernelSize, stride, padding=1, output_padding=0)
#         )
#
#         #region other resolutions
#         # old 64x64
#         # self.network = nn.Sequential(
#         #     nn.Linear(inputSize, depth * 32),
#         #     nn.Unflatten(1, (depth * 32, 1)),
#         #     nn.Unflatten(2, (1, 1)),
#         #     nn.ConvTranspose2d(depth * 32, depth * 4, kernelSize, stride), activation,
#         #     nn.ConvTranspose2d(depth * 4, depth * 2, kernelSize, stride), activation,
#         #     nn.ConvTranspose2d(depth * 2, depth * 1, kernelSize + 1, stride), activation,
#         #     nn.ConvTranspose2d(depth * 1, self.channels, kernelSize + 1, stride))
#
#         # new proposed 64x64
#         # self.network = nn.Sequential(
#         #     nn.Linear(inputSize, 8 * depth * 4 * 4),
#         #     nn.Unflatten(1, (8 * depth, 4, 4)),
#         #     # Input: (256, 4, 4)
#         #     nn.ConvTranspose2d(8 * depth, 4 * depth, kernelSize, stride, padding=1),
#         #     activation,
#         #     # -> (128, 8, 8)
#         #     nn.ConvTranspose2d(4 * depth, 2 * depth, kernelSize, stride, padding=1),
#         #     activation,
#         #     # -> (64, 16, 16)
#         #     nn.ConvTranspose2d(2 * depth, 1 * depth, kernelSize, stride, padding=1),
#         #     activation,
#         #     # -> (32, 32, 32)
#         #     nn.ConvTranspose2d(1 * depth, self.channels, kernelSize, stride, padding=1)
#         #     # -> (3, 64, 64)
#         # )
#         #endregion
#
#     def forward(self, x):
#         return self.network(x)


# NEW DECODER SETUP
# def gn(c): return nn.GroupNorm(32, c)
#
# class ResBlockDown(nn.Module):
#     def __init__(self, in_ch, out_ch, k=3, act=nn.SiLU):
#         super().__init__()
#         self.act  = act(inplace=True)
#         self.conv1= nn.Conv2d(in_ch, out_ch, k, padding=1, bias=False)
#         self.gn1  = gn(out_ch)
#         self.conv2= nn.Conv2d(out_ch, out_ch, k, stride=2, padding=1, bias=False)
#         self.gn2  = gn(out_ch)
#         self.skip = nn.Conv2d(in_ch, out_ch, 1, stride=2, bias=False)
#
#     def forward(self, x):
#         h = self.conv1(x)
#         h = self.act(self.gn1(h))
#         h = self.conv2(h)
#         h = self.gn2(h)
#         s = self.skip(x)
#         return self.act(h + s)
#
# class ResBlockUp(nn.Module):
#     def __init__(self, in_ch, out_ch, k=3, act=nn.SiLU):
#         super().__init__()
#         self.act   = act(inplace=True)
#         self.convT1= nn.ConvTranspose2d(in_ch, out_ch, k, padding=1, bias=False)
#         self.gn1   = gn(out_ch)
#         self.convT2= nn.ConvTranspose2d(out_ch, out_ch, k, stride=2, padding=1, output_padding=1, bias=False)
#         self.gn2   = gn(out_ch)
#         self.skip  = nn.ConvTranspose2d(in_ch, out_ch, 1, stride=2, output_padding=1, bias=False)
#
#     def forward(self, x):
#         h = self.convT1(x)
#         h = self.act(self.gn1(h))
#         h = self.convT2(h)
#         h = self.gn2(h)
#         s = self.skip(x)
#         return self.act(h + s)
#
#
# class EncoderConv(nn.Module):
#     def __init__(self, inputShape, outputSize):
#         super().__init__()
#         self.outputSize = outputSize
#
#         channels, _, _ = inputShape
#         depth   = 32   # base channel multiplier
#         ksize   = 3    # 3×3 inside residual blocks
#
#         self.stem = nn.Sequential(
#             nn.Conv2d(channels, depth, 3, padding=1, bias=False),
#             gn(depth),
#             nn.SiLU(inplace=True),
#         )
#
#         self.down = nn.Sequential(
#             ResBlockDown(depth,      1 * depth, ksize),  # 128 → 64
#             ResBlockDown(1 * depth,  2 * depth, ksize),  # 64  → 32
#             ResBlockDown(2 * depth,  4 * depth, ksize),  # 32  → 16
#             ResBlockDown(4 * depth,  8 * depth, ksize),  # 16  → 8
#             ResBlockDown(8 * depth, 32 * depth, ksize),  # 8   → 4
#         )
#
#         self.head = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(32 * depth * 4 * 4, outputSize),
#             nn.SiLU()
#         )
#
#     def forward(self, x):
#         x = self.stem(x)
#         x = self.down(x)
#         x = self.head(x)
#         return x.view(-1, self.outputSize)
#
# class DecoderConv(nn.Module):
#     def __init__(self, inputSize, outputShape):
#         super().__init__()
#         channels, _, _ = outputShape
#         depth   = 32
#         ksize   = 3
#
#         self.fc = nn.Sequential(
#             nn.Linear(inputSize, depth * 32 * 4 * 4),
#             nn.SiLU()
#         )
#
#         self.unflatten = nn.Unflatten(1, (depth * 32, 4, 4))
#
#         self.up = nn.Sequential(
#             ResBlockUp(32 * depth,  8 * depth, ksize),   # 4 → 8
#             ResBlockUp( 8 * depth,  4 * depth, ksize),   # 8 → 16
#             ResBlockUp( 4 * depth,  2 * depth, ksize),   # 16→ 32
#             ResBlockUp( 2 * depth,  1 * depth, ksize),   # 32→ 64
#         )
#
#         # final RGB reconstruction (no norm, SiLU optional)
#         self.tail = nn.ConvTranspose2d(depth, channels, 4, stride=2, padding=1)
#
#     def forward(self, x):
#         x = self.fc(x)
#         x = self.unflatten(x)
#         x = self.up(x)
#         x = self.tail(x)
#         return x


# NEW GPT5 SETUP
# ---------- Fast MMO Encoder (old topology, depth configurable) ----------
class EncoderConv(nn.Module):
    def __init__(self, inputShape, outputSize):
        super().__init__()
        self.outputSize = outputSize
        channels, _, _ = inputShape
        kernelSize = 4
        stride = 2
        depth = 32  # try 16 first on Mac; bump to 32 if you have headroom

        act = nn.ReLU(inplace=True)
        self.net = nn.Sequential(
            nn.Conv2d(channels,   1 * depth, kernelSize, stride, padding=1, bias=True), act,  # 128→64
            nn.Conv2d(1 * depth,  2 * depth, kernelSize, stride, padding=1, bias=True), act,  # 64 →32
            nn.Conv2d(2 * depth,  4 * depth, kernelSize, stride, padding=1, bias=True), act,  # 32 →16
            nn.Conv2d(4 * depth,  8 * depth, kernelSize, stride, padding=1, bias=True), act,  # 16 →8
            nn.Conv2d(8 * depth, 32 * depth, kernelSize, stride, padding=1, bias=True), act,  # 8  →4
            nn.Flatten(),
            nn.Linear(32 * depth * 4 * 4, outputSize),
            nn.SiLU()  # slightly nicer head nonlinearity
        )

    def forward(self, x):
        x = x.contiguous(memory_format=torch.channels_last)
        return self.net(x).view(-1, self.outputSize)

# ---------- Fast MMO Decoder (Upsample + Conv is faster on MPS) ----------
class _UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.seq(x)

class DecoderConv(nn.Module):
    def __init__(self, inputSize, outputShape):
        super().__init__()
        channels, _, _ = outputShape
        depth = 32  # must match encoder’s setting

        self.fc = nn.Sequential(
            nn.Linear(inputSize, depth * 32 * 4 * 4),
            nn.SiLU()
        )
        self.unflatten = nn.Unflatten(1, (depth * 32, 4, 4))
        self.recon_log_std = nn.Parameter(torch.tensor(-1.4))
        self.up1 = _UpBlock(32 * depth, 8 * depth)   # 4→8
        self.up2 = _UpBlock( 8 * depth, 4 * depth)   # 8→16
        self.up3 = _UpBlock( 4 * depth, 2 * depth)   # 16→32
        self.up4 = _UpBlock( 2 * depth, 1 * depth)   # 32→64
        self.up5 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)  # 64→128
        self.out = nn.Conv2d(1 * depth, channels, 3, padding=1, bias=True)

    def forward(self, x):
        x = self.fc(x)
        x = self.unflatten(x)
        x = x.contiguous(memory_format=torch.channels_last)
        x = self.up1(x); x = self.up2(x); x = self.up3(x); x = self.up4(x)
        x = self.up5(x)
        x = self.out(x)
        return x