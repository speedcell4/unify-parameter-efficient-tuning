#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import os

from torchdevice import set_cuda_visible_devices

num_devices = int(os.environ.get('NUM_DEVICES', 0))
if num_devices > 0:
    set_cuda_visible_devices(n=num_devices)

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset, load_metric
from filelock import FileLock

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

sys.path.insert(2, "./")

from petl.options import (
    GenerationArguments,
    TuneArguments,
)
from petl.petl_encdec_model import PETLEncDecModel

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.9.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

import re
from re import Pattern

import torch
from einops import rearrange
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from transformers import PreTrainedModel, RobertaModel, BartModel
from transformers.models.bart.modeling_bart import BartEncoder, BartDecoder, BartForConditionalGeneration

TODO_SVD = '_todo_svd'

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


def householder(tensor: Tensor, vec: Tensor) -> Tensor:
    vec = F.normalize(vec, p=2, dim=-1)
    return tensor - (vec * tensor).sum(dim=-1, keepdim=True) * vec * 2


class HouseSvdEmbedding(nn.Module):
    def __init__(self, *, embedding: nn.Embedding, sigma: bool, left: int, right: int) -> None:
        super(HouseSvdEmbedding, self).__init__()

        with torch.no_grad():
            u, s, v = torch.linalg.svd(embedding.weight.data.to(device=device), full_matrices=False)

        self.register_buffer('u', u.clone().detach())
        self.register_buffer('z', s.clone().detach())
        self.register_buffer('v', v.clone().detach())

        self.right = nn.ParameterList([
            nn.Parameter(F.normalize(torch.ones_like(s), p=2, dim=-1), requires_grad=True)
            for _ in range(right)
        ])

        self.s = nn.Parameter(s.clone().detach(), requires_grad=sigma)

        self.left = nn.ParameterList([
            nn.Parameter(F.normalize(torch.ones_like(s), p=2, dim=-1), requires_grad=True)
            for _ in range(left)
        ])

        self.num_embeddings = embedding.num_embeddings
        self.embedding_dim = embedding.embedding_dim
        self.padding_idx = embedding.padding_idx
        self.max_norm = embedding.max_norm
        self.norm_type = embedding.norm_type
        self.scale_grad_by_freq = embedding.scale_grad_by_freq
        self.sparse = embedding.sparse

    def extra_repr(self) -> str:
        return ', '.join([
            f'{self.num_embeddings}',
            f'{self.embedding_dim}',
            f'padding_idx={self.padding_idx}',
            f'sigma={self.s.requires_grad}',
        ])

    def forward(self, tensor: Tensor) -> Tensor:
        if self.training or getattr(self, TODO_SVD, True):
            weight = self.u

            for vec in self.left:
                weight = householder(weight, vec=vec)
            weight = weight * self.s
            for vec in self.right:
                weight = householder(weight, vec=vec)

            self.weight = F.linear(weight, weight=self.v.t())
            setattr(self, TODO_SVD, self.training)

        return F.embedding(
            weight=self.weight, input=tensor, padding_idx=self.padding_idx,
            max_norm=self.max_norm, norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq, sparse=self.sparse
        )


class ForwardEmbedding(nn.Module):
    def __init__(self, *, embedding: HouseSvdEmbedding) -> None:
        super(ForwardEmbedding, self).__init__()

        self.embedding = embedding

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.embedding.extra_repr()})'

    def forward(self, tensor: Tensor) -> Tensor:
        return self.embedding(tensor)


class TiedLinear(nn.Module):
    def __init__(self, *, embedding: HouseSvdEmbedding) -> None:
        super(TiedLinear, self).__init__()

        self.embedding = embedding

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.embedding.extra_repr()})'

    def forward(self, tensor: Tensor) -> Tensor:
        return F.linear(tensor, weight=self.embedding.weight)


class HouseMultiHeadReadLinear(nn.Module):
    def __init__(self, *, linear: nn.Linear, sigma: bool, num_heads: int, in_or_left: int, out_or_right: int) -> None:
        super(HouseMultiHeadReadLinear, self).__init__()

        with torch.no_grad():
            u, s, v = torch.linalg.svd(
                rearrange(linear.weight.data.to(device=device), '(h y) x -> h y x', h=num_heads),
                full_matrices=False,
            )

        self.register_buffer('u', u.clone().detach())
        self.register_buffer('z', s.clone().detach())
        self.register_buffer('v', v.clone().detach())

        self.outside = nn.ParameterList([
            nn.Parameter(F.normalize(torch.ones_like(s), p=2, dim=-1), requires_grad=True)
            for _ in range(out_or_right)
        ])

        self.s = nn.Parameter(s.clone().detach(), requires_grad=sigma)

        self.inside = nn.ParameterList([
            nn.Parameter(F.normalize(torch.ones_like(s), p=2, dim=-1), requires_grad=True)
            for _ in range(in_or_left)
        ])

        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.bias = linear.bias
        self.num_heads = num_heads

    def extra_repr(self) -> str:
        return ', '.join([
            f'{self.in_features}',
            f'{self.out_features}',
            f'bias={self.bias is not None}',
            f'num_heads={self.num_heads}',
        ])

    def _forward_old(self, tensor: Tensor) -> Tensor:
        tensor = torch.einsum('...x,hyx->...hy', tensor, self.v)

        for vec in self.outside:
            tensor = householder(tensor, vec=vec)
        tensor = tensor * self.s
        for vec in self.inside:
            tensor = householder(tensor, vec=vec)

        tensor = torch.einsum('...hy,hzy->...hz', tensor, self.u).flatten(start_dim=-2)
        if self.bias is not None:
            tensor = tensor + self.bias
        return tensor

    def forward(self, tensor: Tensor) -> Tensor:
        if self.training or getattr(self, TODO_SVD, True):
            weight = self.u

            for vec in self.inside:
                weight = householder(weight, vec=vec[:, None, :])
            weight = weight * self.s[..., None, :]
            for vec in self.outside:
                weight = householder(weight, vec=vec[:, None, :])

            self.weight = (weight @ self.v).flatten(end_dim=1)
            setattr(self, TODO_SVD, self.training)

        return F.linear(tensor, weight=self.weight, bias=self.bias)


class HouseMultiHeadWriteLinear(nn.Module):
    def __init__(self, *, linear: nn.Linear, sigma: bool, num_heads: int,
                 out_or_left: int, in_or_right: int) -> None:
        super(HouseMultiHeadWriteLinear, self).__init__()

        with torch.no_grad():
            u, s, v = torch.linalg.svd(
                rearrange(linear.weight.data.to(device=device), 'y (h x) -> h y x', h=num_heads),
                full_matrices=False,
            )

        self.register_buffer('u', u.clone().detach())
        self.register_buffer('z', s.clone().detach())
        self.register_buffer('v', v.clone().detach())

        self.inside = nn.ParameterList([
            nn.Parameter(F.normalize(torch.ones_like(s), p=2, dim=-1), requires_grad=True)
            for _ in range(in_or_right)
        ])

        self.s = nn.Parameter(s.clone().detach(), requires_grad=sigma)

        self.outside = nn.ParameterList([
            nn.Parameter(F.normalize(torch.ones_like(s), p=2, dim=-1), requires_grad=True)
            for _ in range(out_or_left)
        ])

        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.bias = linear.bias
        self.num_heads = num_heads

    def extra_repr(self) -> str:
        return ', '.join([
            f'{self.in_features}',
            f'{self.out_features}',
            f'bias={self.bias is not None}',
            f'num_heads={self.num_heads}',
        ])

    def _forward_old(self, tensor: Tensor) -> Tensor:
        tensor = tensor.view((*tensor.size()[:-1], self.num_heads, -1))
        tensor = torch.einsum('...hx,hyx->...hy', tensor, self.v)

        for vec in self.inside:
            tensor = householder(tensor, vec=vec)
        tensor = tensor * self.s
        for vec in self.outside:
            tensor = householder(tensor, vec=vec)

        tensor = torch.einsum('...hy,hzy->...z', tensor, self.u)
        if self.bias is not None:
            tensor = tensor + self.bias
        return tensor

    def forward(self, tensor: Tensor) -> Tensor:
        if self.training or getattr(self, TODO_SVD, True):
            weight = self.u

            for vec in self.outside:
                weight = householder(weight, vec=vec[:, None, :])
            weight = weight * self.s[..., None, :]
            for vec in self.inside:
                weight = householder(weight, vec=vec[:, None, :])

            self.weight = (weight @ self.v).transpose(0, 1).flatten(start_dim=-2)
            setattr(self, TODO_SVD, self.training)

        return F.linear(tensor, weight=self.weight, bias=self.bias)


class ResSvdEmbedding(nn.Module):
    def __init__(self, *, embedding: nn.Embedding, sigma: bool, left: int, right: int) -> None:
        super(ResSvdEmbedding, self).__init__()

        with torch.no_grad():
            u, s, v = torch.linalg.svd(embedding.weight.data.to(device=device), full_matrices=False)

        self.register_buffer('u', u.clone().detach())
        self.register_buffer('z', s.clone().detach())
        self.register_buffer('v', v.clone().detach())

        self.right = nn.ParameterList([
            nn.Parameter(F.normalize(torch.ones_like(s), p=2, dim=-1), requires_grad=True)
            for _ in range(right)
        ])

        self.s = nn.Parameter(torch.zeros_like(s), requires_grad=sigma)

        self.left = nn.ParameterList([
            nn.Parameter(F.normalize(torch.ones_like(s), p=2, dim=-1), requires_grad=True)
            for _ in range(left)
        ])

        self.num_embeddings = embedding.num_embeddings
        self.embedding_dim = embedding.embedding_dim
        self.padding_idx = embedding.padding_idx
        self.max_norm = embedding.max_norm
        self.norm_type = embedding.norm_type
        self.scale_grad_by_freq = embedding.scale_grad_by_freq
        self.sparse = embedding.sparse

    def extra_repr(self) -> str:
        return ', '.join([
            f'{self.num_embeddings}',
            f'{self.embedding_dim}',
            f'padding_idx={self.padding_idx}',
            f'sigma={self.s.requires_grad}',
        ])

    def forward(self, tensor: Tensor) -> Tensor:
        if self.training or getattr(self, TODO_SVD, True):
            weight = res = self.u

            for vec in self.left:
                weight = householder(weight, vec=vec)
            weight = weight * self.z
            for vec in self.right:
                weight = householder(weight, vec=vec)

            weight = weight + res * self.s
            self.weight = F.linear(weight, weight=self.v.t())
            setattr(self, TODO_SVD, self.training)

        return F.embedding(
            weight=self.weight, input=tensor, padding_idx=self.padding_idx,
            max_norm=self.max_norm, norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq, sparse=self.sparse
        )


class ResMultiHeadReadLinear(nn.Module):
    def __init__(self, *, linear: nn.Linear, sigma: bool, num_heads: int, in_or_left: int, out_or_right: int) -> None:
        super(ResMultiHeadReadLinear, self).__init__()

        with torch.no_grad():
            u, s, v = torch.linalg.svd(
                rearrange(linear.weight.data.to(device=device), '(h y) x -> h y x', h=num_heads),
                full_matrices=False,
            )

        self.register_buffer('u', u.clone().detach())
        self.register_buffer('z', s.clone().detach())
        self.register_buffer('v', v.clone().detach())

        self.outside = nn.ParameterList([
            nn.Parameter(F.normalize(torch.ones_like(s), p=2, dim=-1), requires_grad=True)
            for _ in range(out_or_right)
        ])

        self.s = nn.Parameter(torch.zeros_like(s), requires_grad=sigma)

        self.inside = nn.ParameterList([
            nn.Parameter(F.normalize(torch.ones_like(s), p=2, dim=-1), requires_grad=True)
            for _ in range(in_or_left)
        ])

        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.bias = linear.bias
        self.num_heads = num_heads
        self.weight = {}

    def extra_repr(self) -> str:
        return ', '.join([
            f'{self.in_features}',
            f'{self.out_features}',
            f'bias={self.bias is not None}',
            f'num_heads={self.num_heads}',
        ])

    def _forward_old(self, tensor: Tensor) -> Tensor:
        tensor = res = torch.einsum('...x,hyx->...hy', tensor, self.v)

        for vec in self.outside:
            tensor = householder(tensor, vec=vec)
        tensor = tensor * self.z
        for vec in self.inside:
            tensor = householder(tensor, vec=vec)

        tensor = tensor + res * self.s
        tensor = torch.einsum('...hy,hzy->...hz', tensor, self.u).flatten(start_dim=-2)
        if self.bias is not None:
            tensor = tensor + self.bias
        return tensor

    def forward(self, tensor: Tensor) -> Tensor:
        if self.training or getattr(self, TODO_SVD, True):
            weight = res = self.u

            for vec in self.inside:
                weight = householder(weight, vec=vec[:, None, :])
            weight = weight * self.z[..., None, :]
            for vec in self.outside:
                weight = householder(weight, vec=vec[:, None, :])

            weight = weight + res * self.s[..., None, :]
            self.weight[tensor.device] = (weight @ self.v).flatten(end_dim=1)
            setattr(self, TODO_SVD, self.training)

        return F.linear(tensor, weight=self.weight[tensor.device], bias=self.bias)


class ResMultiHeadWriteLinear(nn.Module):
    def __init__(self, *, linear: nn.Linear, sigma: bool, num_heads: int,
                 out_or_left: int, in_or_right: int) -> None:
        super(ResMultiHeadWriteLinear, self).__init__()

        with torch.no_grad():
            u, s, v = torch.linalg.svd(
                rearrange(linear.weight.data.to(device=device), 'y (h x) -> h y x', h=num_heads),
                full_matrices=False,
            )

        self.register_buffer('u', u.clone().detach())
        self.register_buffer('z', s.clone().detach())
        self.register_buffer('v', v.clone().detach())

        self.inside = nn.ParameterList([
            nn.Parameter(F.normalize(torch.ones_like(s), p=2, dim=-1), requires_grad=True)
            for _ in range(in_or_right)
        ])

        self.s = nn.Parameter(torch.zeros_like(s), requires_grad=sigma)

        self.outside = nn.ParameterList([
            nn.Parameter(F.normalize(torch.ones_like(s), p=2, dim=-1), requires_grad=True)
            for _ in range(out_or_left)
        ])

        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.bias = linear.bias
        self.num_heads = num_heads
        self.weight = {}

    def extra_repr(self) -> str:
        return ', '.join([
            f'{self.in_features}',
            f'{self.out_features}',
            f'bias={self.bias is not None}',
            f'num_heads={self.num_heads}',
        ])

    def _forward_old(self, tensor: Tensor) -> Tensor:
        tensor = tensor.view((*tensor.size()[:-1], self.num_heads, -1))
        tensor = res = torch.einsum('...hx,hyx->...hy', tensor, self.v)

        for vec in self.inside:
            tensor = householder(tensor, vec=vec)
        tensor = tensor * self.z
        for vec in self.outside:
            tensor = householder(tensor, vec=vec)

        tensor = tensor + res * self.s
        tensor = torch.einsum('...hy,hzy->...z', tensor, self.u)
        if self.bias is not None:
            tensor = tensor + self.bias
        return tensor

    def forward(self, tensor: Tensor) -> Tensor:
        if self.training or getattr(self, TODO_SVD, True):
            weight = res = self.u

            for vec in self.outside:
                weight = householder(weight, vec=vec[:, None, :])
            weight = weight * self.z[..., None, :]
            for vec in self.inside:
                weight = householder(weight, vec=vec[:, None, :])

            weight = weight + res * self.s[..., None, :]
            self.weight[tensor.device] = (weight @ self.v).transpose(0, 1).flatten(start_dim=-2)
            setattr(self, TODO_SVD, self.training)

        return F.linear(tensor, weight=self.weight[tensor.device], bias=self.bias)


# if __name__ == '__main__':
#     layer1 = nn.Embedding(100, 768)
#     layer2 = HouseSvdEmbedding(embedding=layer1, sigma=True, left=0, right=0)
#     layer3 = TiedLinear(embedding=layer2)
#     x = torch.arange(100)
#     print(torch.allclose(layer1(x), layer2(x), atol=1e-5))
#
#     x = torch.randn((5, 768))
#     excepted = F.linear(x, weight=layer1.weight)
#     actual = layer3(x)
#
#     print(torch.allclose(actual, excepted, atol=1e-4))
#
#     layer1 = nn.Linear(768, 768)
#     layer2 = HouseMultiHeadWriteLinear(
#         linear=layer1, sigma=True, num_heads=1,
#         out_or_left=2, in_or_right=0,
#     )
#     x = torch.randn((100, 768))
#     # print(torch.dist(layer1(x), layer2(x)))
#     print(torch.allclose(layer1(x), layer2._forward_old(x), atol=1e-5))
#     print(torch.allclose(layer2(x), layer2._forward_old(x), atol=1e-5))
#
#     layer1 = nn.Linear(768, 768)
#     layer2 = HouseMultiHeadReadLinear(
#         linear=layer1, sigma=True, num_heads=4,
#         in_or_left=2, out_or_right=0,
#     )
#     x = torch.randn((100, 768))
#     # print(torch.dist(layer1(x), layer2(x)))
#     print(torch.allclose(layer1(x), layer2._forward_old(x), atol=1e-5))
#     print(torch.allclose(layer2(x), layer2._forward_old(x), atol=1e-5))

ROBERTA_PATTERNS = {
    'w': re.compile(r'word_embeddings'),
    'p': re.compile(r'position_embeddings'),
    't': re.compile(r'token_type_embeddings'),
    'q': re.compile(r'query'),
    'k': re.compile(r'key'),
    'v': re.compile(r'value'),
    'a': re.compile(r'attention\.output\.dense'),
    'i': re.compile(r'intermediate\.dense'),
    'o': re.compile(r'\d+\.output\.dense'),
    'l': re.compile(r'attention\.output\.LayerNorm'),
    'n': re.compile(r'\d+\.output\.LayerNorm'),
}

BART_PATTERNS = {
    'w': re.compile(r'shared'),
    'p': re.compile(r'embed_positions'),
    't': re.compile(r'token_type_embeddings'),
    'q': re.compile(r'q_proj'),
    'k': re.compile(r'k_proj'),
    'v': re.compile(r'v_proj'),
    'a': re.compile(r'out_proj'),
    'i': re.compile(r'fc1'),
    'o': re.compile(r'fc2'),
    'l': re.compile(r'self_attn_layer_norm'),
    'n': re.compile(r'final_layer_norm'),
}


def contain(pattern: Pattern, string: str) -> bool:
    return re.search(pattern=pattern, string=string) is not None


def _house(keys: str = 'qiao',
           sigma: bool = True, bias: bool = True,
           inside: int = 1, outside: int = 1, size: int = 1, *,
           model: PreTrainedModel, **kwargs):
    model.requires_grad_(False)

    if isinstance(model, RobertaModel):
        pattern_map = ROBERTA_PATTERNS
    elif isinstance(model, (BartModel, BartEncoder, BartDecoder)):
        pattern_map = BART_PATTERNS
    else:
        raise NotImplementedError
    patterns = tuple(pattern_map[key] for key in keys)

    mods = {name: module for name, module in model.named_modules()}

    for name, mod in tqdm(list(mods.items()), desc=f'preparing {house.__name__}-ing'):
        if any(re.search(pattern=pattern, string=name) is not None for pattern in patterns):
            parent, y = name.rsplit('.', maxsplit=1)

            if isinstance(mod, nn.Embedding):
                if contain(pattern_map['w'], name) or contain(pattern_map['p'], name):
                    setattr(mods[parent], y, HouseSvdEmbedding(
                        embedding=mod, sigma=sigma, left=inside, right=outside,
                    ))

                elif contain(pattern_map['t'], name):
                    mod.requires_grad_(bias)

            elif isinstance(mod, nn.Linear):
                if contain(pattern_map['q'], name) or contain(pattern_map['k'], name) or contain(pattern_map['v'],
                                                                                                 name):
                    svd_mod = HouseMultiHeadReadLinear(
                        linear=mod, sigma=sigma, in_or_left=inside, out_or_right=outside,
                        num_heads=model.config.num_attention_heads,
                    )
                elif contain(pattern_map['a'], name):
                    svd_mod = HouseMultiHeadWriteLinear(
                        linear=mod, sigma=sigma, out_or_left=outside, in_or_right=inside,
                        num_heads=model.config.num_attention_heads,
                    )
                elif contain(pattern_map['i'], name):
                    svd_mod = HouseMultiHeadReadLinear(
                        linear=mod, sigma=sigma, in_or_left=inside, out_or_right=outside,
                        num_heads=size,
                    )
                elif contain(pattern_map['o'], name):
                    svd_mod = HouseMultiHeadWriteLinear(
                        linear=mod, sigma=sigma, out_or_left=outside, in_or_right=inside,
                        num_heads=size,
                    )
                else:
                    raise NotImplementedError

                svd_mod.bias.requires_grad_(bias)
                setattr(mods[parent], y, svd_mod)

            elif isinstance(mod, nn.LayerNorm):
                mod.weight.requires_grad_(sigma)
                mod.bias.requires_grad_(bias)

            else:
                raise TypeError(f'type of {name} is {type(mod)}, not supported yet')


def _res(keys: str = 'qiao',
         sigma: bool = True, bias: bool = True,
         inside: int = 1, outside: int = 1, size: int = 1, *,
         model: PreTrainedModel, **kwargs):
    model.requires_grad_(False)

    if isinstance(model, RobertaModel):
        pattern_map = ROBERTA_PATTERNS
    elif isinstance(model, (BartModel, BartEncoder, BartDecoder)):
        pattern_map = BART_PATTERNS
    else:
        raise NotImplementedError
    patterns = tuple(pattern_map[key] for key in keys)

    mods = {name: module for name, module in model.named_modules()}

    for name, mod in tqdm(list(mods.items()), desc=f'preparing {house.__name__}-ing'):
        if any(re.search(pattern=pattern, string=name) is not None for pattern in patterns):
            parent, y = name.rsplit('.', maxsplit=1)

            if isinstance(mod, nn.Embedding):
                if contain(pattern_map['w'], name) or contain(pattern_map['p'], name):
                    setattr(mods[parent], y, ResSvdEmbedding(
                        embedding=mod, sigma=sigma, left=inside, right=outside,
                    ))

                elif contain(pattern_map['t'], name):
                    mod.requires_grad_(bias)

            elif isinstance(mod, nn.Linear):
                if contain(pattern_map['q'], name) or contain(pattern_map['k'], name) or contain(pattern_map['v'],
                                                                                                 name):
                    svd_mod = ResMultiHeadReadLinear(
                        linear=mod, sigma=sigma, in_or_left=inside, out_or_right=outside,
                        num_heads=model.config.num_attention_heads,
                    )
                elif contain(pattern_map['a'], name):
                    svd_mod = ResMultiHeadWriteLinear(
                        linear=mod, sigma=sigma, out_or_left=outside, in_or_right=inside,
                        num_heads=model.config.num_attention_heads,
                    )
                elif contain(pattern_map['i'], name):
                    svd_mod = ResMultiHeadReadLinear(
                        linear=mod, sigma=sigma, in_or_left=inside, out_or_right=outside,
                        num_heads=size,
                    )
                elif contain(pattern_map['o'], name):
                    svd_mod = ResMultiHeadWriteLinear(
                        linear=mod, sigma=sigma, out_or_left=outside, in_or_right=inside,
                        num_heads=size,
                    )
                else:
                    raise NotImplementedError

                svd_mod.bias.requires_grad_(bias)
                setattr(mods[parent], y, svd_mod)

            elif isinstance(mod, nn.LayerNorm):
                mod.weight.requires_grad_(sigma)
                mod.bias.requires_grad_(bias)

            else:
                raise TypeError(f'type of {name} is {type(mod)}, not supported yet')


def house(keys: str = 'qiao', size: int = 1, inside: int = 0, outside: int = 2,
          embed: bool = True, enc: bool = True, dec: bool = True, *,
          model: BartForConditionalGeneration, **kwargs):
    model.requires_grad_(False)

    if enc:
        _house(keys=keys, size=size, model=model.model.encoder, inside=inside, outside=outside)
        # if embed:
        #     model.model.encoder.embed_positions = HouseSvdEmbedding(
        #         embedding=model.model.encoder.embed_positions, sigma=True, left=inside, right=outside,
        #     )

    if dec:
        _house(keys=keys, size=size, model=model.model.decoder, inside=inside, outside=outside)
        # if embed:
        #     model.model.decoder.embed_positions = HouseSvdEmbedding(
        #         embedding=model.model.decoder.embed_positions, sigma=True, left=inside, right=outside,
        #     )

    if embed:
        embedding = HouseSvdEmbedding(embedding=model.model.shared, sigma=True, left=inside, right=outside)
        model.model.shared = embedding
        model.model.encoder.embed_tokens = embedding
        model.model.decoder.embed_tokens = ForwardEmbedding(embedding=embedding)
        model.lm_head = TiedLinear(embedding=embedding)


def res(keys: str = 'qiao', size: int = 1, inside: int = 0, outside: int = 2,
        embed: bool = True, enc: bool = True, dec: bool = True, *,
        model: BartForConditionalGeneration, **kwargs):
    model.requires_grad_(False)

    if enc:
        _res(keys=keys, size=size, model=model.model.encoder, inside=inside, outside=outside)
        # if embed:
        #     model.model.encoder.embed_positions = HouseSvdEmbedding(
        #         embedding=model.model.encoder.embed_positions, sigma=True, left=inside, right=outside,
        #     )

    if dec:
        _res(keys=keys, size=size, model=model.model.decoder, inside=inside, outside=outside)
        # if embed:
        #     model.model.decoder.embed_positions = HouseSvdEmbedding(
        #         embedding=model.model.decoder.embed_positions, sigma=True, left=inside, right=outside,
        #     )

    if embed:
        embedding = HouseSvdEmbedding(embedding=model.model.shared, sigma=True, left=inside, right=outside)
        model.model.shared = embedding
        model.model.encoder.embed_tokens = embedding
        model.model.decoder.embed_tokens = ForwardEmbedding(embedding=embedding)
        model.lm_head = TiedLinear(embedding=embedding)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge) on "
                    "(a jsonlines or csv file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on " "(a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                    "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                    "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                    "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    max_tokens_per_batch: Optional[int] = field(
        default=0,
        metadata={
            "help": "dynamic batching. Override batch size when larger than 0"
        },
    )
    num_rotations: int = field(
        default=2,
        metadata={
            'help': 'how many rotations',
        }
    )
    decompose_embed: bool = field(
        default=False,
        metadata={
            'help': 'decompose embedding',
        }
    )
    use_res: bool = field(
        default=False,
        metadata={
            'help': 'use res',
        }
    )
    fc_size: int = field(
        default=1,
        metadata={
            'help': 'ffn size',
        }
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
}


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments,
         GenerationArguments, TuneArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, gen_args, tune_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, gen_args, tune_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if training_args.should_log else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    logger.info(f"Training/evaluation parameters {training_args}")

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the full texts and the second column for the
    # summaries (unless you specify column names for this with the `text_column` and `summary_column` arguments).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir)
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # put generation args into config
    for k, v in vars(gen_args).items():
        setattr(config, f'gen_{k}', v)

    try:
        attn_gate = float(tune_args.attn_gate)
        tune_args.attn_gate = attn_gate
    except:
        pass

    try:
        ffn_gate = float(tune_args.ffn_gate)
        tune_args.ffn_gate = ffn_gate
    except:
        pass

    # put useful args into config: these arguments will be used in models, thus adding them to config
    # interested_args = ['use_prefix', 'mid_dim', 'preseqlen', 'prefix_dropout', 'unfreeze_params']
    for k, v in vars(tune_args).items():
        if not hasattr(config, k):
            setattr(config, k, v)

    for k in ['max_source_length', 'max_target_length']:
        setattr(config, k, vars(data_args)[k])

    setattr(training_args, 'max_tokens_per_batch', data_args.max_tokens_per_batch)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # import pdb; pdb.set_trace()
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    elif training_args.do_eval:
        column_names = datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Get the column names for input/target.
    dataset_columns = summarization_name_mapping.get(data_args.dataset_name, None)
    if data_args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = data_args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.summary_column is None:
        summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = data_args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    def preprocess_function(examples):
        inputs = examples[text_column]
        targets = examples[summary_column]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        predict_dataset = predict_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on prediction dataset",
        )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    if data_args.use_res:
        res(keys='qiao', size=data_args.fc_size,
            embed=data_args.decompose_embed, inside=0, outside=data_args.num_rotations, model=model)
    else:
        house(keys='qiao', size=data_args.fc_size,
              embed=data_args.decompose_embed, inside=0, outside=data_args.num_rotations, model=model)

    logger.info(f'model => {model}')

    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f'{name}.size() => {param.size()}')

    # Metric
    metric = load_metric("rouge")

    gen_prefix = "val"

    def postprocess_text(preds, labels):
        str_preds = [pred.strip() for pred in preds]
        str_labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in str_preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in str_labels]

        return preds, labels, str_preds, str_labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels, str_decoded_preds, str_decoded_labels = postprocess_text(decoded_preds,
                                                                                                decoded_labels)

        # only write in the main process
        if trainer.is_world_process_zero():
            fout_pred = open(os.path.join(training_args.output_dir, f"{gen_prefix}.pred.summary"), "w",
                             encoding="utf-8")
            fout_gold = open(os.path.join(training_args.output_dir, f"{gen_prefix}.gold.summary"), "w",
                             encoding="utf-8")
            for pred, gold in zip(str_decoded_preds, str_decoded_labels):
                # print(pred)
                # print(gold)
                fout_pred.write(pred + "\n")
                fout_gold.write(gold + "\n")
            fout_pred.close()
            fout_gold.close()

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(
            max_length=data_args.val_max_target_length, num_beams=gen_args.num_beams, metric_key_prefix="eval"
        )
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        gen_prefix = "test"
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            predict_dataset,
            metric_key_prefix="predict",
            max_length=data_args.val_max_target_length,
            num_beams=gen_args.num_beams,
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w") as writer:
                    writer.write("\n".join(predictions))

    if training_args.push_to_hub:
        kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "summarization"}
        if data_args.dataset_name is not None:
            kwargs["dataset_tags"] = data_args.dataset_name
            if data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = data_args.dataset_config_name
                kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
            else:
                kwargs["dataset"] = data_args.dataset_name

        trainer.push_to_hub(**kwargs)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
