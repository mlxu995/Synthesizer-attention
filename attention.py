import math
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init


# Original dot product self-attention
class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer

    :param int n_head: the number of heads
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate
    """

    def __init__(self, n_head, n_feat, dropout_rate):
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, query, key, value, mask):
        """Compute 'Scaled Dot Product Attention'

        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time2, size)
        :param torch.Tensor value: (batch, time2, size)
        :param torch.Tensor mask: (batch, time1, time2)
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
             weighted by the query dot key attention (batch, head, time1, time2)
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch, head, time1, time2)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
            min_value = float(numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
        return self.linear_out(x)  # (batch, time1, d_model)


class DenseSynthesizerAttention(nn.Module):
    """Dense Synthesizer attention layer

    :param int n_head: the number of heads
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate
    :param int time2: the number of time steps of target
    """

    def __init__(self, n_head, n_feat, dropout_rate, time2):
        super(DenseSynthesizerAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.time2 = time2
        self.linear1 = nn.Linear(n_feat, n_feat)
        self.linear2 = nn.Linear(n_feat, n_head*time2)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, query, key, value, mask):
        """Compute 'Dense Synthesizer Attention'

        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: not use
        :param torch.Tensor value: (batch, time2, size)
        :param torch.Tensor mask: (batch, time1, time2)
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
             weighted by the dense attention (batch, head, time1, time2)
        """
        n_batch = query.size(0)

        B = self.linear2(F.relu(self.linear1(query))).view(n_batch, -1, self.h, self.time2)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        B = B.transpose(1, 2)  # (batch, head, time1, time2)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
            min_value = float(numpy.finfo(torch.tensor(0, dtype=B.dtype).numpy().dtype).min)
            B = B.masked_fill(mask, min_value)
            self.attn = torch.softmax(B, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(B, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
        return self.linear_out(x)  # (batch, time1, d_model)


class RandomSynthesizerAttention(nn.Module):
    """Random Synthesizer attention layer

    :param int n_head: the number of head s
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate
    :param int time1: the number of time steps of self
    :param int time2: the number of time steps of target
    """

    def __init__(self, n_head, n_feat, dropout_rate, time1, time2):
        super(RandomSynthesizerAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.time1 = time1
        self.time2 = time2
        self.attention_weight = Parameter(torch.Tensor(n_head*time1, time2))
        init.kaiming_uniform_(self.attention_weight, a=math.sqrt(5))
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, query, key, value, mask):
        """Compute 'Random Synthesizer Attention'

        :param torch.Tensor query: not use
        :param torch.Tensor key: not use
        :param torch.Tensor value: (batch, time2, size)
        :param torch.Tensor mask: (batch, time1, time2)
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
             weighted by global attention (batch, head, time1, time2)
        """
        n_batch = query.size(0)

        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        self.attn = self.attention_weight.view(1, self.h, self.time1, self.time2)  # (1, head, time1, time2)
        self.attn = self.attn.repeat([n_batch, 1, 1, 1])  # (n_batch, head, time1, time2)

        # Random Synthesizer Attention may not need mask
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
            min_value = float(numpy.finfo(torch.tensor(0, dtype=self.attn.dtype).numpy().dtype).min)
            self.attn = self.attn.masked_fill(mask, min_value)
            self.attn = torch.softmax(self.attn, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(self.attn, dim=-1)  # (batch, head, time1, time2)
        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
        return self.linear_out(x)  # (batch, time1, d_model)


if __name__ == '__main__':

    denseattn = DenseSynthesizerAttention(n_head=2, n_feat=2, dropout_rate=0,
                                    time2=6)

    randomattn = RandomSynthesizerAttention(n_head=2, n_feat=2, dropout_rate=0,
                                    time1=6, time2=6)

    x1 = [
            [
                [1, 2],
                [3, 1],
                [5, 1],
                [7, 2],
                [1, 6],
                [1, 1],
            ],
            [
                [1, 1],
                [2, 1],
                [6, 1],
                [0, 0],
                [0, 0],
                [0, 0],
            ],
            [
                [3, 1],
                [1, 2],
                [6, 3],
                [2, 3],
                [0, 0],
                [0, 0],
            ],
    ]
    x1 = torch.Tensor(x1)

    x2 = torch.randn((3, 6, 2))

    y1 = denseattn(x1, x1, x1, None)
    print(y1)

    y2 = randomattn(x2, x2, x2, None)
    print(y2)

