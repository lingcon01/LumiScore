# dynamic triplet-attentive mechanism
from typing import Union, Optional
from torch_geometric.typing import (Adj, Size, OptTensor, PairTensor)

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from torch_geometric.nn.inits import glorot
from torch import nn

import numpy as np



def get_adj_matrix(adj_matrix, n_nodes, batch_size, device):
    # adj_matrices = torch.cat(tuple(adj_matrix), 1)

    n = len(adj_matrix)
    adj_matrix = torch.tensor(adj_matrix)
    row_all = []
    col_all = []
    for i in range(n):
        # 将邻接矩阵转换为NumPy数组

        # 获取连接边的节点索引
        rows, cols = torch.where(adj_matrix[i] == 1)

        row_all.append(rows + i * n_nodes)
        col_all.append(cols + i * n_nodes)
        # 构建边索引
        # edge_index = torch.tensor([rows+i*n_nodes, cols+i*n_nodes], dtype=torch.long)

        # 将边索引添加到列表中
    row_all = torch.cat(row_all, dim=0).to(device)
    col_all = torch.cat(col_all, dim=0).to(device)

    keep_indices = []
    for i in range(len(row_all)):
        if row_all[i] != col_all[i]:
            keep_indices.append(i)

    row_all = row_all[keep_indices]
    col_all = col_all[keep_indices]

    edges = [row_all, col_all]

    return edges


def get_adj_list(adj_list, n_nodes, batch_size, device):
    # adj_matrices = torch.cat(tuple(adj_matrix), 1)

    n = len(adj_list)
    adj_list = torch.tensor(adj_list)
    row_all = []
    col_all = []
    for i in range(n):
        # 将邻接矩阵转换为NumPy数组

        # 获取连接边的节点索引
        rows, cols = adj_list[i]

        row_all.append(rows + i * n_nodes)
        col_all.append(cols + i * n_nodes)
        # 构建边索引
        # edge_index = torch.tensor([rows+i*n_nodes, cols+i*n_nodes], dtype=torch.long)

        # 将边索引添加到列表中
    row_all = torch.cat(row_all, dim=0).to(device, dtype=torch.long)
    col_all = torch.cat(col_all, dim=0).to(device, dtype=torch.long)

    keep_indices = []
    for i in range(len(row_all)):
        if row_all[i] != col_all[i]:
            keep_indices.append(i)

    row_all = row_all[keep_indices]
    col_all = col_all[keep_indices]

    edges = [row_all, col_all]

    # unique_indices = []
    # for i in range(len(edges[0])):
    #     column = [row[i] for row in edges]
    #     if column.count(column[0]) == len(column):
            # 所有元素都相同，跳过
    #         continue
    #     unique_indices.append(i)

    # 选择不重复的列
    # filtered_edges = [[row[i] for i in unique_indices] for row in edges]

    return edges

def coord2radial(edge_index, coord):
    row, col = edge_index
    coord_diff = coord[row] - coord[col]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)

    return radial, coord_diff


class EdgeGATConv(MessagePassing):
    _alpha: OptTensor

    def __init__(self, in_channels: int,
                 out_channels: int, edge_dim: int, heads: int = 1,
                 negative_slope: float = 0.2, dropout: float = 0.,
                 bias: bool = True, share_weights: bool = False,
                 **kwargs):
        super(EdgeGATConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.share_weights = share_weights

        self.lin_l = Linear(in_channels, heads * out_channels, bias=bias)
        if share_weights:
            self.lin_r = self.lin_l
        else:
            self.lin_r = Linear(in_channels, heads * out_channels, bias=bias)

        self.att = Parameter(torch.Tensor(1, heads, out_channels))

        self.lin_edge = Linear(edge_dim, heads * out_channels, bias=bias)

        self.lin_out = Linear(heads * out_channels, out_channels)

        layer = nn.Linear(out_channels, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(out_channels, out_channels))
        coord_mlp.append(nn.ReLU())
        coord_mlp.append(layer)

        self.coord_mlp = nn.Sequential(*coord_mlp)

        self._alpha = None

        self.edge_feature = None

        self.reset_parameters()

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        trans = torch.clamp(trans, min=-100,
                            max=100)  # This is never activated but just in case it case it explosed it may save the train
        agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        coord += agg * 1.0
        return coord

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        glorot(self.lin_edge.weight)
        glorot(self.lin_out.weight)
        glorot(self.att)

    def forward(self, x, edge_index, edge_attr,
                size: Size = None, return_attention_weights: bool = None):

        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2
            x_l = self.lin_l(x).view(-1, H, C)
            if self.share_weights:
                x_r = x_l
            else:
                x_r = self.lin_r(x).view(-1, H, C)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2
            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None

        edge_attr = self.lin_edge(edge_attr).view(-1, H, C)

        # propagate_type: (x: PairTensor)
        out = self.propagate(edge_index, x=(x_l, x_r), edge_attr=edge_attr, size=size)

        # edge_feature = self.edge_feature.squeeze(1)

        # pos = self.coord_model(atom_positions, edge_index, coord_diff, edge_feature)

        # pos = torch.reshape(pos, (a, n_nodes, 3))

        alpha = self._alpha
        self._alpha = None

        out = out.view(-1, self.heads * self.out_channels)
        out = self.lin_out(out)

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, x_i: Tensor, edge_attr,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        x = x_i + x_j + edge_attr

        # x = x_i + x_j

        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        edge = x_j * edge_attr * alpha.unsqueeze(-1)

        self.edge_feature = x

        return edge


