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


def householder(tensor: Tensor, vec: Tensor) -> Tensor:
    vec = F.normalize(vec, p=2, dim=-1)
    return tensor - (vec * tensor).sum(dim=-1, keepdim=True) * vec * 2


class HouseSvdEmbedding(nn.Module):
    def __init__(self, *, embedding: nn.Embedding, sigma: bool, left: int, right: int) -> None:
        super(HouseSvdEmbedding, self).__init__()

        with torch.no_grad():
            u, s, v = torch.linalg.svd(embedding.weight.data, full_matrices=False)

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
        self.weight = {}

    def extra_repr(self) -> str:
        return ', '.join([
            f'{self.num_embeddings}',
            f'{self.embedding_dim}',
            f'padding_idx={self.padding_idx}',
            f'sigma={self.s.requires_grad}',
        ])

    def forward(self, tensor: Tensor) -> Tensor:
        weight = F.embedding(
            weight=self.u, input=tensor, padding_idx=self.padding_idx,
            max_norm=self.max_norm, norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq, sparse=self.sparse
        )

        for vec in self.left:
            weight = householder(weight, vec=vec)
        weight = weight * self.s
        for vec in self.right:
            weight = householder(weight, vec=vec)

        return weight @ self.v


class TiedLinear(nn.Module):
    def __init__(self, *, embedding: HouseSvdEmbedding) -> None:
        super(TiedLinear, self).__init__()

        self.embedding = embedding

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.extra_repr()})'

    def extra_repr(self) -> str:
        return ', '.join([
            f'{self.embedding.num_embeddings}',
            f'{self.embedding.embedding_dim}',
            f'{self.embedding.padding_idx}',
        ])

    def forward(self, tensor: Tensor) -> Tensor:
        tensor = F.linear(tensor, weight=self.embedding.v)

        for vec in self.embedding.right:
            tensor = householder(tensor, vec=vec)
        tensor = tensor * self.embedding.s
        for vec in self.embedding.left:
            tensor = householder(tensor, vec=vec)

        return F.linear(tensor, weight=self.embedding.u)


class HouseMultiHeadReadLinear(nn.Module):
    def __init__(self, *, linear: nn.Linear, sigma: bool, num_heads: int, in_or_left: int, out_or_right: int) -> None:
        super(HouseMultiHeadReadLinear, self).__init__()

        with torch.no_grad():
            u, s, v = torch.linalg.svd(
                rearrange(linear.weight.data, '(h y) x -> h y x', h=num_heads),
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
        self.weight = {}

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

            self.weight[tensor.device] = (weight @ self.v).flatten(end_dim=1)
            setattr(self, TODO_SVD, self.training)

        return F.linear(tensor, weight=self.weight[tensor.device], bias=self.bias)


class HouseMultiHeadWriteLinear(nn.Module):
    def __init__(self, *, linear: nn.Linear, sigma: bool, num_heads: int,
                 out_or_left: int, in_or_right: int) -> None:
        super(HouseMultiHeadWriteLinear, self).__init__()

        with torch.no_grad():
            u, s, v = torch.linalg.svd(
                rearrange(linear.weight.data, 'y (h x) -> h y x', h=num_heads),
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

            self.weight[tensor.device] = (weight @ self.v).transpose(0, 1).flatten(start_dim=-2)
            setattr(self, TODO_SVD, self.training)

        return F.linear(tensor, weight=self.weight[tensor.device], bias=self.bias)


if __name__ == '__main__':
    layer1 = nn.Embedding(100, 768)
    layer2 = HouseSvdEmbedding(embedding=layer1, sigma=True, left=0, right=0)
    layer3 = TiedLinear(embedding=layer2)
    x = torch.arange(100)
    print(torch.allclose(layer1(x), layer2(x), atol=1e-5))

    x = torch.randn((5, 768))
    excepted = F.linear(x, weight=layer1.weight)
    actual = layer3(x)

    print(torch.allclose(actual, excepted, atol=1e-4))

    layer1 = nn.Linear(768, 768)
    layer2 = HouseMultiHeadWriteLinear(
        linear=layer1, sigma=True, num_heads=1,
        out_or_left=2, in_or_right=0,
    )
    x = torch.randn((100, 768))
    # print(torch.dist(layer1(x), layer2(x)))
    print(torch.allclose(layer1(x), layer2._forward_old(x), atol=1e-5))
    print(torch.allclose(layer2(x), layer2._forward_old(x), atol=1e-5))

    layer1 = nn.Linear(768, 768)
    layer2 = HouseMultiHeadReadLinear(
        linear=layer1, sigma=True, num_heads=4,
        in_or_left=2, out_or_right=0,
    )
    x = torch.randn((100, 768))
    # print(torch.dist(layer1(x), layer2(x)))
    print(torch.allclose(layer1(x), layer2._forward_old(x), atol=1e-5))
    print(torch.allclose(layer2(x), layer2._forward_old(x), atol=1e-5))

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
        model.model.set_input_embeddings(embedding)
        model.model.encoder.set_input_embeddings(embedding)
        model.model.decoder.set_input_embeddings(embedding)
        model.lm_head = TiedLinear(embedding=embedding)
