import torch_geometric
import torch
from torch import nn
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import dense_to_sparse
import numpy as np


class TransformerConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, head=1, depth=6):
        super(TransformerConvLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.head = head
        self.depth = depth
        for i in range(depth):
            attn = TransformerConv(in_dim[i], out_dim[i], heads=head, edge_dim=in_dim[0])
            self.add_module(f'attn_{i}', attn)


    def forward(self, node, edge, A):
        """
        node: b n 1
        edge: b n n
        A: b n n
        n1: b
        n2: b
        """
        b, n, _ = node.shape
        node_attr = torch.zeros((b, n, self.out_dim[-1]))
        for i in range(b):
            val = dense_to_sparse(edge[i] * A[i])
            emb = node[i]
            for j in range(self.depth):
                attn = getattr(self, f'attn_{j}')
                emb = attn(emb, edge_index=val[0], edge_attr=val[1].unsqueeze(-1))
            node_attr[i] = emb

        return node_attr.to(node.device)