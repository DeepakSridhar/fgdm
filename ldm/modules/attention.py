from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

from ldm.modules.diffusionmodules.util import checkpoint


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
    
    def reshape_batch_dim_to_heads_and_average(self, tensor):
        batch_size, seq_len, seq_len2 = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size,
                                head_size, seq_len, seq_len2)
        return tensor.mean(1)

    def forward(self, x, context=None, adapt=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        # if adapt is not None:
            # v += adapt

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)
        attention_probs_avg = self.reshape_batch_dim_to_heads_and_average(
            sim)

        out = einsum('b i j, b j d -> b i d', attn, v)
        if adapt is not None:
            q = rearrange(adapt, 'b n (h d) -> (b h) n d', h=h)

            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            attention_probs_avg_adapt = self.reshape_batch_dim_to_heads_and_average(
                sim)

            out_adapt = einsum('b i j, b j d -> b i d', attn, v)
            out += out_adapt
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out), attention_probs_avg


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None, adapt=None):
        return checkpoint(self._forward, (x, context, adapt), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None, adapt=None):
        y1 = self.attn1(self.norm1(x))
        x = y1[0] + x
        y2 = self.attn2(self.norm2(x), context=context, adapt=adapt)
        x = y2[0] + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context=None, adapt=None):
        if isinstance(x, tuple):
            x = x[0]
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        # y1s = []
        # y2s = []
        for block in self.transformer_blocks:
            x = block(x, context=context, adapt=adapt)
            # y1s.append(y1)
            # y2s.append(y2)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in


class CrossAttentionM(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
    
    def reshape_batch_dim_to_heads_and_average(self, tensor):
        batch_size, seq_len, seq_len2 = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size,
                                head_size, seq_len, seq_len2)
        return tensor.mean(1)

    def forward(self, x, context=None, adapt=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        # if adapt is not None:
            # v += adapt

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)
        attention_probs_avg = self.reshape_batch_dim_to_heads_and_average(
            sim)

        out = einsum('b i j, b j d -> b i d', attn, v)
        if adapt is not None:
            q = rearrange(adapt, 'b n (h d) -> (b h) n d', h=h)

            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            attention_probs_avg_adapt = self.reshape_batch_dim_to_heads_and_average(
                sim)

            out_adapt = einsum('b i j, b j d -> b i d', attn, v)
            out += out_adapt
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out), attn



class SelfAttentionM(nn.Module):
    def __init__(self, query_dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout) )

    def forward(self, x, gated=False):
        q = self.to_q(x) # B*N*(H*C)
        k = self.to_k(x) # B*N*(H*C)
        v = self.to_v(x) # B*N*(H*C)

        B, N, HC = q.shape 
        H = self.heads
        C = HC // H 
        # if gated: import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        q = q.view(B,N,H,C).permute(0,2,1,3).reshape(B*H,N,C) # (B*H)*N*C
        k = k.view(B,N,H,C).permute(0,2,1,3).reshape(B*H,N,C) # (B*H)*N*C
        v = v.view(B,N,H,C).permute(0,2,1,3).reshape(B*H,N,C) # (B*H)*N*C

        sim = torch.einsum('b i c, b j c -> b i j', q, k) * self.scale  # (B*H)*N*N
        attn = sim.softmax(dim=-1) # (B*H)*N*N
       
        out = torch.einsum('b i j, b j c -> b i c', attn, v) # (B*H)*N*C
        out = out.view(B,H,N,C).permute(0,2,1,3).reshape(B,N,(H*C)) # B*N*(H*C)
        return self.to_out(out), attn
       

    
    def visual_att(self, att):
        global iter_att
        ll = [0,2,7]
        for i in range():
            kk = torch.sum(att[i],axis=0)
            kk = kk[:4096].reshape(64,64)
            save_image( (kk-kk.min()) / (kk.max() - kk.min()) , os.path.join('att', str(iter_att) + '_' +str(i) + '.png'))
        iter_att += 1


class BasicTransformerBlockM(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttentionM(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttentionM(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None, adapt=None):
        return checkpoint(self._forward, (x, context, adapt), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None, adapt=None):
        y1, self_prob = self.attn1(self.norm1(x))
        x = y1 + x
        y2, prob = self.attn2(self.norm2(x), context=context, adapt=adapt)
        x = y2 + x
        x = self.ff(self.norm3(x)) + x
        return x, prob, self_prob, None 
        

class SpatialTransformerM(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlockM(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context=None, adapt=None):
        if isinstance(x, tuple):
            x = x[0]
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        probs = [] 
        self_prob_list = []
        self_prob_grounding_list = []
        for block in self.transformer_blocks:
            x, prob, self_prob, grouding_prob = block(x, context, adapt=adapt)
            if prob is not None: probs.append(prob)
            if self_prob is not None: self_prob_list.append(self_prob)
            if grouding_prob is not None: self_prob_grounding_list.append(grouding_prob)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in, probs, self_prob_list, self_prob_grounding_list