import torch as th
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter_add
from torch_geometric.utils import to_dense_batch
from torch import nn
from .layer.egat import EdgeGATConv
from GenScore.model.layer.subgraph import subgraph
from .transform_AK.transform import SubgraphsTransform
from .transform_AK.config import cfg, update_cfg
import math
from .layers import InteractionNet
import logging
import math
from torch.distributions import Normal

from .transform_AK.element import MLP, DiscreteEncoder, Identity, VNUpdate


def glorot_orthogonal(tensor, scale):
    """Initialize a tensor's values according to an orthogonal Glorot initialization scheme."""
    if tensor is not None:
        th.nn.init.orthogonal_(tensor.data)
        scale /= ((tensor.size(-2) + tensor.size(-1)) * tensor.var())
        tensor.data *= scale.sqrt()


class BiEGCL(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    """

    def __init__(self, in_dim, hid_dim, node_attr_dim=0, edge_attr_dim=0, activation="ReLU",
                 residual=True, gated=False, normalize=False, coords_agg='mean', tanh=False,
                 coord_change_maximum=10):
        super(BiEGCL, self).__init__()
        self.residual = residual
        self.gated = gated
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.coord_change_maximum = coord_change_maximum
        in_edge_dim = in_dim * 2
        edge_coor_dim = 1

        if isinstance(activation, str):
            self.activation = getattr(nn, activation)()
        else:
            self.activation = activation

        self.edge_mlp_s2t = nn.Sequential(
            nn.Linear(in_edge_dim + edge_coor_dim + edge_attr_dim, hid_dim),
            self.activation,
            nn.Linear(hid_dim, hid_dim),
            self.activation
        )

        self.edge_mlp_t2s = nn.Sequential(
            nn.Linear(in_edge_dim + edge_coor_dim + edge_attr_dim, hid_dim),
            self.activation,
            nn.Linear(hid_dim, hid_dim),
            self.activation
        )
        if self.gated:
            self.gate_mlp_s2t = nn.Sequential(
                nn.Linear(hid_dim, 1),
                nn.Sigmoid()
            )
            self.gate_mlp_t2s = nn.Sequential(
                nn.Linear(hid_dim, 1),
                nn.Sigmoid()
            )

        self.node_mlp_s = nn.Sequential(
            nn.Linear(hid_dim + in_dim + node_attr_dim, hid_dim),
            self.activation,
            nn.Linear(hid_dim, hid_dim)
        )

        self.node_mlp_t = nn.Sequential(
            nn.Linear(hid_dim + in_dim + node_attr_dim, hid_dim),
            self.activation,
            nn.Linear(hid_dim, hid_dim)
        )

        weight_layer = nn.Linear(hid_dim, 1, bias=False)
        th.nn.init.xavier_uniform_(weight_layer.weight, gain=0.001)
        if self.tanh:
            self.coord_mlp = nn.Sequential(
                nn.Linear(hid_dim, hid_dim),
                self.activation,
                weight_layer,
                nn.Tanh()
            )
        else:
            self.coord_mlp = nn.Sequential(
                nn.Linear(hid_dim, hid_dim),
                self.activation,
                weight_layer
            )

        if self.coords_agg == "attention":
            # self.attent_mlp = nn.Linear(hid_dim, 1) # TODO: refine
            self.attent_mlp_s2t = nn.Linear(hid_dim, 1)
            self.attent_mlp_t2s = nn.Linear(hid_dim, 1)

    def unsorted_segment_sum(self, data, segment_ids, num_segments):
        result_shape = (num_segments, data.size(1))
        result = data.new_full(result_shape, 0)
        segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
        result.scatter_add_(0, segment_ids, data)

        return result

    def unsorted_segment_mean(self, data, segment_ids, num_segments):
        result_shape = (num_segments, data.size(1))
        result = data.new_full(result_shape, 0)
        count = data.new_full(result_shape, 0)
        segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
        result.scatter_add_(0, segment_ids, data)
        count.scatter_add_(0, segment_ids, th.ones_like(data))

        return result / count.clamp(min=1)

    def edge_function(self, edge_src_feat, edge_tgt_feat, radial, edge_attr):
        if edge_attr is None:
            edge_feat_in = th.cat([edge_src_feat, edge_tgt_feat, radial], dim=1)
        else:
            edge_feat_in = th.cat([edge_src_feat, edge_tgt_feat, radial, edge_attr], dim=1)
        edge_feat_out_s2t = self.edge_mlp_s2t(edge_feat_in)
        edge_feat_out_t2s = self.edge_mlp_t2s(edge_feat_in)
        if self.gated:
            edge_feat_out_s2t = edge_feat_out_s2t * self.gate_mlp_s2t(edge_feat_out_s2t)
            edge_feat_out_t2s = edge_feat_out_t2s * self.gate_mlp_t2s(edge_feat_out_t2s)

        return edge_feat_out_s2t, edge_feat_out_t2s

    def attention_function(self, edge_feat_s2t, edge_feat_t2s, edge_list):
        edge_src = edge_list[0]
        edge_tgt = edge_list[1]
        attent_score_s2t = self.attent_mlp_s2t(edge_feat_s2t)
        attent_weight_s2t = scatter_softmax(attent_score_s2t.squeeze(), index=edge_tgt).unsqueeze(-1)

        attent_score_t2s = self.attent_mlp_t2s(edge_feat_t2s)
        attent_weight_t2s = scatter_softmax(attent_score_t2s.squeeze(), index=edge_src).unsqueeze(-1)

        return attent_weight_s2t, attent_weight_t2s

    def tgt_coord_function(self, tgt_node_coord, edge_list, coord_diff, edge_feat_s2t,
                           attent_weight_s2t=None):
        # Action on Node Out
        edge_src = edge_list[0]
        edge_tgt = edge_list[1]
        weighted_trans = coord_diff * self.coord_mlp(edge_feat_s2t)

        if self.coords_agg == 'sum':
            agg_trans = self.unsorted_segment_sum(weighted_trans, edge_tgt, num_segments=tgt_node_coord.size(0))
        elif self.coords_agg == 'mean':
            agg_trans = self.unsorted_segment_mean(weighted_trans, edge_tgt, num_segments=tgt_node_coord.size(0))
        elif self.coords_agg == "attention":
            weighted_trans = weighted_trans * attent_weight_s2t
            agg_trans = self.unsorted_segment_sum(weighted_trans, edge_tgt, num_segments=tgt_node_coord.size(0))
        else:
            raise NotImplementedError('Aggregation method {} is not implemented'.format(self.coords_agg))
        tgt_node_coord += agg_trans.clamp(-self.coord_change_maximum, self.coord_change_maximum)

        return tgt_node_coord

    def node_function(self, edge_list, src_node_feat, tgt_node_feat, edge_feat_s2t, edge_feat_t2s,
                      attent_weight_s2t=None, attent_weight_t2s=None):
        edge_src = edge_list[0]
        edge_tgt = edge_list[1]
        # TODO: Attention
        if self.coords_agg == "attention":
            edge_feat_s2t = edge_feat_s2t * attent_weight_s2t
            edge_feat_t2s = edge_feat_t2s * attent_weight_t2s
            agg_edge_feat_s2t = self.unsorted_segment_sum(edge_feat_s2t, edge_tgt, num_segments=tgt_node_feat.size(0))
            agg_edge_feat_t2s = self.unsorted_segment_sum(edge_feat_t2s, edge_src, num_segments=src_node_feat.size(0))
        else:
            agg_edge_feat_s2t = self.unsorted_segment_sum(edge_feat_s2t, edge_tgt, num_segments=tgt_node_feat.size(0))
            agg_edge_feat_t2s = self.unsorted_segment_sum(edge_feat_t2s, edge_src, num_segments=src_node_feat.size(0))

        tgt_node_feat_in = th.cat([tgt_node_feat, agg_edge_feat_s2t], dim=1)
        src_node_feat_in = th.cat([src_node_feat, agg_edge_feat_t2s], dim=1)

        tgt_node_feat_out = self.node_mlp_t(tgt_node_feat_in)
        src_node_feat_out = self.node_mlp_s(src_node_feat_in)
        if self.residual:
            if tgt_node_feat.size(1) == tgt_node_feat_out.size(1):
                tgt_node_feat_out = tgt_node_feat + tgt_node_feat_out
            if src_node_feat.size(1) == src_node_feat_out.size(1):
                src_node_feat_out = src_node_feat + src_node_feat_out
        return tgt_node_feat_out, src_node_feat_out

    def coord2radial(self, edge_list, src_node_coord, tgt_node_coord, epsilon=1e-6):
        edge_src = edge_list[0]
        edge_tgt = edge_list[1]
        coord_diff = tgt_node_coord[edge_tgt] - src_node_coord[edge_src]
        radial = th.sum(coord_diff ** 2, 1).unsqueeze(1)

        if self.normalize:
            norm = th.sqrt(radial).detach() + epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, src_node_feat, tgt_node_feat, src_node_coord, tgt_node_coord,
                edge_list, edge_attr=None, output_coord=False):
        radial, coord_diff = self.coord2radial(edge_list, src_node_coord, tgt_node_coord)

        edge_feat_s2t, edge_feat_t2s = self.edge_function(src_node_feat[edge_list[0]], tgt_node_feat[edge_list[1]],
                                                          radial, edge_attr)
        if self.coords_agg == "attention":
            attent_weight_s2t, attent_weight_t2s = self.attention_function(edge_feat_s2t, edge_feat_t2s, edge_list)
            if output_coord:
                tgt_node_coord = self.tgt_coord_function(tgt_node_coord, edge_list, coord_diff, edge_feat_s2t,
                                                         attent_weight_s2t=attent_weight_s2t)
            tgt_node_feat_out, src_node_feat_out = self.node_function(edge_list, src_node_feat, tgt_node_feat,
                                                                      edge_feat_s2t, edge_feat_t2s,
                                                                      attent_weight_s2t=attent_weight_s2t,
                                                                      attent_weight_t2s=attent_weight_t2s)
        else:
            if output_coord:
                tgt_node_coord = self.tgt_coord_function(tgt_node_coord, edge_list, coord_diff, edge_feat_s2t)
            tgt_node_feat_out, src_node_feat_out = self.node_function(edge_list, src_node_feat, tgt_node_feat,
                                                                      edge_feat_s2t, edge_feat_t2s)
        if output_coord:
            return src_node_feat_out, tgt_node_feat_out, src_node_coord, tgt_node_coord
        else:
            return src_node_feat_out, tgt_node_feat_out


class coords_update(nn.Module):
    def __init__(self, dim_dh, num_head, drop_rate=0.15):
        super().__init__()
        self.num_head = num_head
        self.attention2deltax = nn.Sequential(
            nn.Linear(dim_dh, dim_dh // 2),
            nn.Dropout(p=drop_rate),
            nn.LeakyReLU(),
            nn.Linear(dim_dh // 2, self.num_head)
        )
        self.weighted_head_layer = nn.Linear(num_head, 1, bias=False)

    def forward(self, a_ij, coords, edge_index):
        i, j = edge_index
        delta_x = coords[i] - coords[j]
        delta_x = delta_x / (th.norm(delta_x, p=2, dim=-1).unsqueeze(dim=-1) + 1e-6)
        delta_x = delta_x * self.weighted_head_layer(self.attention2deltax(a_ij))
        delta_x = scatter(delta_x, index=i, reduce='sum', dim=0)
        coords += delta_x
        return coords


class MultiHeadAttentionLayer(nn.Module):
    """Compute attention scores with a DGLGraph's node and edge (geometric) features."""

    def __init__(self, num_input_feats, num_output_feats,
                 num_heads, using_bias=False, update_edge_feats=True, update_coords=False):
        super(MultiHeadAttentionLayer, self).__init__()

        # Declare shared variables
        self.num_output_feats = num_output_feats
        self.num_heads = num_heads
        self.using_bias = using_bias
        self.update_edge_feats = update_edge_feats
        self.update_coords = update_coords

        # Define node features' query, key, and value tensors, and define edge features' projection tensors
        self.Q = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)
        self.K = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)
        self.V = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)
        self.edge_feats_projection = nn.Linear(num_input_feats + 1, self.num_output_feats * self.num_heads,
                                               bias=using_bias)

        self.coords_update = coords_update(self.num_output_feats * self.num_heads, self.num_heads)

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        scale = 2.0
        if self.using_bias:
            glorot_orthogonal(self.Q.weight, scale=scale)
            self.Q.bias.data.fill_(0)

            glorot_orthogonal(self.K.weight, scale=scale)
            self.K.bias.data.fill_(0)

            glorot_orthogonal(self.V.weight, scale=scale)
            self.V.bias.data.fill_(0)

            glorot_orthogonal(self.edge_feats_projection.weight, scale=scale)
            self.edge_feats_projection.bias.data.fill_(0)
        else:
            glorot_orthogonal(self.Q.weight, scale=scale)
            glorot_orthogonal(self.K.weight, scale=scale)
            glorot_orthogonal(self.V.weight, scale=scale)
            glorot_orthogonal(self.edge_feats_projection.weight, scale=scale)

    def propagate_attention(self, edge_index, node_feats_q, node_feats_k, node_feats_v, edge_feats_projection, coords):
        row, col = edge_index
        e_out = None
        # Compute attention scores
        alpha = node_feats_k[row] * node_feats_q[col]
        # Scale and clip attention scores
        alpha = (alpha / np.sqrt(self.num_output_feats)).clamp(-5.0, 5.0)
        # Use available edge features to modify the attention scores
        alpha = alpha * edge_feats_projection
        # Copy edge features as e_out to be passed to edge_feats_MLP
        if self.update_edge_feats:
            e_out = alpha

            if self.update_coords:
                coords = self.coords_update(alpha, coords, edge_index)

        # Apply softmax to attention scores, followed by clipping
        alphax = th.exp((alpha.sum(-1, keepdim=True)).clamp(-5.0, 5.0))
        # Send weighted values to target nodes
        wV = scatter_add(node_feats_v[row] * alphax, col, dim=0, dim_size=node_feats_q.size(0))
        z = scatter_add(alphax, col, dim=0, dim_size=node_feats_q.size(0))
        return wV, z, e_out, coords

    def forward(self, x, edge_attr, edge_index, coords):
        row, col = edge_index
        node_feats_q = self.Q(x).view(-1, self.num_heads, self.num_output_feats)
        node_feats_k = self.K(x).view(-1, self.num_heads, self.num_output_feats)
        node_feats_v = self.V(x).view(-1, self.num_heads, self.num_output_feats)
        edge_attr = th.cat([edge_attr, th.norm(coords[row] - coords[col], p=2, dim=-1, keepdim=True) * 0.1], dim=-1)
        edge_feats_projection = self.edge_feats_projection(edge_attr).view(-1, self.num_heads, self.num_output_feats)
        wV, z, e_out, coords = self.propagate_attention(edge_index, node_feats_q, node_feats_k, node_feats_v,
                                                        edge_feats_projection, coords)

        h_out = wV / (z + th.full_like(z, 1e-6))

        return h_out, e_out, coords


class GraphTransformerModule(nn.Module):
    """A Graph Transformer module (equivalent to one layer of graph convolutions)."""

    def __init__(
            self,
            num_hidden_channels,
            activ_fn=nn.SiLU(),
            residual=True,
            num_attention_heads=4,
            norm_to_apply='batch',
            dropout_rate=0.1,
            num_layers=4,
    ):
        super(GraphTransformerModule, self).__init__()

        # Record parameters given
        self.activ_fn = activ_fn
        self.residual = residual
        self.num_attention_heads = num_attention_heads
        self.norm_to_apply = norm_to_apply
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers

        # --------------------
        # Transformer Module
        # --------------------
        # Define all modules related to a Geometric Transformer module
        self.apply_layer_norm = 'layer' in self.norm_to_apply.lower()

        self.num_hidden_channels, self.num_output_feats = num_hidden_channels, num_hidden_channels
        if self.apply_layer_norm:
            self.layer_norm1_node_feats = nn.LayerNorm(self.num_output_feats)
            self.layer_norm1_edge_feats = nn.LayerNorm(self.num_output_feats)
        else:  # Otherwise, default to using batch normalization
            self.batch_norm1_node_feats = nn.BatchNorm1d(self.num_output_feats)
            self.batch_norm1_edge_feats = nn.BatchNorm1d(self.num_output_feats)

        self.mha_module = MultiHeadAttentionLayer(
            self.num_hidden_channels,
            self.num_output_feats // self.num_attention_heads,
            self.num_attention_heads,
            self.num_hidden_channels != self.num_output_feats,  # Only use bias if a Linear() has to change sizes
            update_edge_feats=True
        )

        self.O_node_feats = nn.Linear(self.num_output_feats, self.num_output_feats)
        self.O_edge_feats = nn.Linear(self.num_output_feats, self.num_output_feats)

        # MLP for node features
        dropout = nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0.0 else nn.Identity()
        self.node_feats_MLP = nn.ModuleList([
            nn.Linear(self.num_output_feats, self.num_output_feats * 2, bias=False),
            self.activ_fn,
            dropout,
            nn.Linear(self.num_output_feats * 2, self.num_output_feats, bias=False)
        ])

        if self.apply_layer_norm:
            self.layer_norm2_node_feats = nn.LayerNorm(self.num_output_feats)
            self.layer_norm2_edge_feats = nn.LayerNorm(self.num_output_feats)
        else:  # Otherwise, default to using batch normalization
            self.batch_norm2_node_feats = nn.BatchNorm1d(self.num_output_feats)
            self.batch_norm2_edge_feats = nn.BatchNorm1d(self.num_output_feats)

        # MLP for edge features
        self.edge_feats_MLP = nn.ModuleList([
            nn.Linear(self.num_output_feats, self.num_output_feats * 2, bias=False),
            self.activ_fn,
            dropout,
            nn.Linear(self.num_output_feats * 2, self.num_output_feats, bias=False)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        scale = 2.0
        glorot_orthogonal(self.O_node_feats.weight, scale=scale)
        self.O_node_feats.bias.data.fill_(0)
        glorot_orthogonal(self.O_edge_feats.weight, scale=scale)
        self.O_edge_feats.bias.data.fill_(0)

        for layer in self.node_feats_MLP:
            if hasattr(layer, 'weight'):  # Skip initialization for activation functions
                glorot_orthogonal(layer.weight, scale=scale)

        for layer in self.edge_feats_MLP:
            if hasattr(layer, 'weight'):
                glorot_orthogonal(layer.weight, scale=scale)

    def run_gt_layer(self, data, node_feats, edge_feats):
        """Perform a forward pass of geometric attention using a multi-head attention (MHA) module."""
        node_feats_in1 = node_feats  # Cache node representations for first residual connection
        edge_feats_in1 = edge_feats  # Cache edge representations for first residual connection

        # Apply first round of normalization before applying geometric attention, for performance enhancement
        if self.apply_layer_norm:
            node_feats = self.layer_norm1_node_feats(node_feats)
            edge_feats = self.layer_norm1_edge_feats(edge_feats)
        else:  # Otherwise, default to using batch normalization
            node_feats = self.batch_norm1_node_feats(node_feats)
            edge_feats = self.batch_norm1_edge_feats(edge_feats)

        # Get multi-head attention output using provided node and edge representations
        node_attn_out, edge_attn_out, data.pos = self.mha_module(node_feats, edge_feats, data.edge_index, data.pos)

        node_feats = node_attn_out.view(-1, self.num_output_feats)
        edge_feats = edge_attn_out.view(-1, self.num_output_feats)

        node_feats = F.dropout(node_feats, self.dropout_rate, training=self.training)
        edge_feats = F.dropout(edge_feats, self.dropout_rate, training=self.training)

        node_feats = self.O_node_feats(node_feats)
        edge_feats = self.O_edge_feats(edge_feats)

        # Make first residual connection
        if self.residual:
            node_feats = node_feats_in1 + node_feats  # Make first node residual connection
            edge_feats = edge_feats_in1 + edge_feats  # Make first edge residual connection

        node_feats_in2 = node_feats  # Cache node representations for second residual connection
        edge_feats_in2 = edge_feats  # Cache edge representations for second residual connection

        # Apply second round of normalization after first residual connection has been made
        if self.apply_layer_norm:
            node_feats = self.layer_norm2_node_feats(node_feats)
            edge_feats = self.layer_norm2_edge_feats(edge_feats)
        else:  # Otherwise, default to using batch normalization
            node_feats = self.batch_norm2_node_feats(node_feats)
            edge_feats = self.batch_norm2_edge_feats(edge_feats)

        # Apply MLPs for node and edge features
        for layer in self.node_feats_MLP:
            node_feats = layer(node_feats)
        for layer in self.edge_feats_MLP:
            edge_feats = layer(edge_feats)

        # Make second residual connection
        if self.residual:
            node_feats = node_feats_in2 + node_feats  # Make second node residual connection
            edge_feats = edge_feats_in2 + edge_feats  # Make second edge residual connection

        # Return edge representations along with node representations (for tasks other than interface prediction)
        return node_feats, edge_feats

    def forward(self, data, node_feats, edge_feats):
        """Perform a forward pass of a Geometric Transformer to get intermediate node and edge representations."""
        node_feats, edge_feats = self.run_gt_layer(data, node_feats, edge_feats)
        return node_feats, edge_feats


class FinalGraphTransformerModule(nn.Module):
    """A (final layer) Graph Transformer module that combines node and edge representations using self-attention."""

    def __init__(self,
                 num_hidden_channels,
                 activ_fn=nn.SiLU(),
                 residual=True,
                 num_attention_heads=4,
                 norm_to_apply='batch',
                 dropout_rate=0.1,
                 num_layers=4):
        super(FinalGraphTransformerModule, self).__init__()

        # Record parameters given
        self.activ_fn = activ_fn
        self.residual = residual
        self.num_attention_heads = num_attention_heads
        self.norm_to_apply = norm_to_apply
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers

        # --------------------
        # Transformer Module
        # --------------------
        # Define all modules related to a Geometric Transformer module
        self.apply_layer_norm = 'layer' in self.norm_to_apply.lower()

        self.num_hidden_channels, self.num_output_feats = num_hidden_channels, num_hidden_channels
        if self.apply_layer_norm:
            self.layer_norm1_node_feats = nn.LayerNorm(self.num_output_feats)
            self.layer_norm1_edge_feats = nn.LayerNorm(self.num_output_feats)
        else:  # Otherwise, default to using batch normalization
            self.batch_norm1_node_feats = nn.BatchNorm1d(self.num_output_feats)
            self.batch_norm1_edge_feats = nn.BatchNorm1d(self.num_output_feats)

        self.mha_module = MultiHeadAttentionLayer(
            self.num_hidden_channels,
            self.num_output_feats // self.num_attention_heads,
            self.num_attention_heads,
            self.num_hidden_channels != self.num_output_feats,  # Only use bias if a Linear() has to change sizes
            update_edge_feats=False)

        self.O_node_feats = nn.Linear(self.num_output_feats, self.num_output_feats)

        # MLP for node features
        dropout = nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0.0 else nn.Identity()
        self.node_feats_MLP = nn.ModuleList([
            nn.Linear(self.num_output_feats, self.num_output_feats * 2, bias=False),
            self.activ_fn,
            dropout,
            nn.Linear(self.num_output_feats * 2, self.num_output_feats, bias=False)
        ])

        if self.apply_layer_norm:
            self.layer_norm2_node_feats = nn.LayerNorm(self.num_output_feats)
        else:  # Otherwise, default to using batch normalization
            self.batch_norm2_node_feats = nn.BatchNorm1d(self.num_output_feats)

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        scale = 2.0
        glorot_orthogonal(self.O_node_feats.weight, scale=scale)
        self.O_node_feats.bias.data.fill_(0)

        for layer in self.node_feats_MLP:
            if hasattr(layer, 'weight'):  # Skip initialization for activation functions
                glorot_orthogonal(layer.weight, scale=scale)

    # glorot_orthogonal(self.conformation_module.weight, scale=scale)

    def run_gt_layer(self, data, node_feats, edge_feats):
        """Perform a forward pass of geometric attention using a multi-head attention (MHA) module."""
        node_feats_in1 = node_feats  # Cache node representations for first residual connection
        # edge_feats = self.conformation_module(edge_feats)

        # Apply first round of normalization before applying geometric attention, for performance enhancement
        if self.apply_layer_norm:
            node_feats = self.layer_norm1_node_feats(node_feats)
            edge_feats = self.layer_norm1_edge_feats(edge_feats)
        else:  # Otherwise, default to using batch normalization
            node_feats = self.batch_norm1_node_feats(node_feats)
            edge_feats = self.batch_norm1_edge_feats(edge_feats)

        # Get multi-head attention output using provided node and edge representations
        node_attn_out, _, data.pos = self.mha_module(node_feats, edge_feats, data.edge_index, data.pos)
        node_feats = node_attn_out.view(-1, self.num_output_feats)
        node_feats = F.dropout(node_feats, self.dropout_rate, training=self.training)
        node_feats = self.O_node_feats(node_feats)

        # Make first residual connection
        if self.residual:
            node_feats = node_feats_in1 + node_feats  # Make first node residual connection

        node_feats_in2 = node_feats  # Cache node representations for second residual connection

        # Apply second round of normalization after first residual connection has been made
        if self.apply_layer_norm:
            node_feats = self.layer_norm2_node_feats(node_feats)
        else:  # Otherwise, default to using batch normalization
            node_feats = self.batch_norm2_node_feats(node_feats)

        # Apply MLP for node features
        for layer in self.node_feats_MLP:
            node_feats = layer(node_feats)

        # Make second residual connection
        if self.residual:
            node_feats = node_feats_in2 + node_feats  # Make second node residual connection

        # Return node representations
        return node_feats

    def forward(self, data, node_feats, edge_feats):
        """Perform a forward pass of a Geometric Transformer to get final node representations."""
        node_feats = self.run_gt_layer(data, node_feats, edge_feats)
        return node_feats


class GraphTransformer(nn.Module):
    """A graph transformer
	"""

    def __init__(
            self,
            in_channels,
            edge_features=10,
            num_hidden_channels=128,
            activ_fn=nn.SiLU(),
            transformer_residual=True,
            num_attention_heads=4,
            norm_to_apply='batch',
            dropout_rate=0.1,
            num_layers=4,
            **kwargs
    ):
        super(GraphTransformer, self).__init__()

        # Initialize model parameters
        self.activ_fn = activ_fn
        self.transformer_residual = transformer_residual
        self.num_attention_heads = num_attention_heads
        self.norm_to_apply = norm_to_apply
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers

        # --------------------
        # Initializer Modules
        # --------------------
        # Define all modules related to edge and node initialization
        self.node_encoder = nn.Linear(in_channels, num_hidden_channels)
        self.edge_encoder = nn.Linear(edge_features, num_hidden_channels)
        # --------------------
        # Transformer Module
        # --------------------
        # Define all modules related to a variable number of Geometric Transformer modules
        num_intermediate_layers = max(0, num_layers - 1)
        gt_block_modules = [GraphTransformerModule(
            num_hidden_channels=num_hidden_channels,
            activ_fn=activ_fn,
            residual=transformer_residual,
            num_attention_heads=num_attention_heads,
            norm_to_apply=norm_to_apply,
            dropout_rate=dropout_rate,
            num_layers=num_layers) for _ in range(num_intermediate_layers)]
        if num_layers > 0:
            gt_block_modules.extend([FinalGraphTransformerModule(
                num_hidden_channels=num_hidden_channels,
                activ_fn=activ_fn,
                residual=transformer_residual,
                num_attention_heads=num_attention_heads,
                norm_to_apply=norm_to_apply,
                dropout_rate=dropout_rate,
                num_layers=num_layers)])
        self.gt_block = nn.ModuleList(gt_block_modules)

    def forward(self, data):
        node_feats = self.node_encoder(data.x)
        edge_feats = self.edge_encoder(data.edge_attr[:, 1:])

        # Apply a given number of intermediate geometric attention layers to the node and edge features given
        for gt_layer in self.gt_block[:-1]:
            node_feats, edge_feats = gt_layer(data, node_feats, edge_feats)

        # Apply final layer to update node representations by merging current node and edge representations
        node_feats = self.gt_block[-1](data, node_feats, edge_feats)
        data.x = node_feats
        data.edge_attr = edge_feats
        # return node_feats
        return data

class SubGT(nn.Module):
    def __init__(self,
                 in_channels,
                 edge_features=10,
                 num_hidden_channels=128,
                 activ_fn=nn.SiLU(),
                 transformer_residual=True,
                 num_attention_heads=4,
                 norm_to_apply='batch',
                 dropout_rate=0.1,
                 num_layers=4,
                 **kwargs
                 ):
        super(SubGT, self).__init__()

        # Initialize model parameters
        self.activ_fn = activ_fn
        self.transformer_residual = transformer_residual
        self.num_attention_heads = num_attention_heads
        self.norm_to_apply = norm_to_apply
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers

        # --------------------
        # Initializer Modules
        # --------------------
        # Define all modules related to edge and node initialization
        self.node_encoder = nn.Linear(in_channels, num_hidden_channels)
        self.edge_encoder = nn.Linear(edge_features, num_hidden_channels)
        # --------------------
        # Transformer Module
        # --------------------
        # Define all modules related to a variable number of Geometric Transformer modules
        num_intermediate_layers = max(0, num_layers - 1)
        gt_block_modules = [GraphTransformerModule(
            num_hidden_channels=num_hidden_channels,
            activ_fn=activ_fn,
            residual=transformer_residual,
            num_attention_heads=num_attention_heads,
            norm_to_apply=norm_to_apply,
            dropout_rate=dropout_rate,
            num_layers=num_layers) for _ in range(num_intermediate_layers)]
        if num_layers > 0:
            gt_block_modules.extend([FinalGraphTransformerModule(
                num_hidden_channels=num_hidden_channels,
                activ_fn=activ_fn,
                residual=transformer_residual,
                num_attention_heads=num_attention_heads,
                norm_to_apply=norm_to_apply,
                dropout_rate=dropout_rate,
                num_layers=num_layers)])
        self.gt_block = nn.ModuleList(gt_block_modules)

        self.transform = SubgraphsTransform(cfg.subgraph.hops,
                                            walk_length=cfg.subgraph.walk_length,
                                            p=cfg.subgraph.walk_p,
                                            q=cfg.subgraph.walk_q,
                                            repeat=cfg.subgraph.walk_repeat,
                                            sampling_mode=cfg.sampling.mode,
                                            minimum_redundancy=cfg.sampling.redundancy,
                                            shortest_path_mode_stride=cfg.sampling.stride,
                                            random_mode_sampling_rate=cfg.sampling.random_rate,
                                            random_init=True)

        self.transform_eval = SubgraphsTransform(cfg.subgraph.hops,
                                                 walk_length=cfg.subgraph.walk_length,
                                                 p=cfg.subgraph.walk_p,
                                                 q=cfg.subgraph.walk_q,
                                                 repeat=cfg.subgraph.walk_repeat,
                                                 sampling_mode=None,
                                                 random_init=False)

        self.sub_GT = subgraph(self.gt_block, num_hidden_channels)

    def forward(self, data, model_type=None):
        edge_index = data.edge_index
        data.x = self.node_encoder(data.x)
        data.edge_attr = self.edge_encoder(data.edge_attr[:, :10])

        sample = dict()

        sample['subgraphs_batch'], sample['subgraphs_nodes_mapper'], sample['subgraphs_edges_mapper'], sample[
            'combined_subgraphs'], sample['hop_indicator'], sample['num_nodes'] = self.transform_eval(data.edge_index,
                                                                                                      data.x.size()[0])

        # Apply a given number of intermediate geometric attention layers to the node and edge features given

        data.x = self.sub_GT(data, sample, model_type)
        data.edge_index = edge_index

        return data


logging.basicConfig(filename='output.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ==============================================
class GenScore(nn.Module):
    def __init__(self, ligand_model, target_model, in_channels, hidden_dim, n_gaussians, dropout_rate=0.15,
                 dist_threhold=1000):
        super(GenScore, self).__init__()

        self.dropout_rate = dropout_rate
        self.ligand_model = ligand_model
        self.target_model = target_model
        self.interact_block = nn.ModuleList([BiEGCL(in_dim=128, hid_dim=128, edge_attr_dim=0,
                                                    activation="SiLU", residual=True, gated=True, normalize=False,
                                                    coords_agg='mean',
                                                    coord_change_maximum=10) for _ in range(2)])

        # print(self.interact_block)

        self.MLP = nn.Sequential(nn.Linear(in_channels * 2, hidden_dim),
                                 nn.BatchNorm1d(hidden_dim),
                                 nn.ELU(),
                                 nn.Dropout(p=dropout_rate))

        self.z_pi = nn.Linear(hidden_dim, n_gaussians)
        self.z_sigma = nn.Linear(hidden_dim, n_gaussians)
        self.z_mu = nn.Linear(hidden_dim, n_gaussians)
        self.atom_types = nn.Linear(in_channels, 17)
        self.bond_types = nn.Linear(in_channels * 2, 4)

        # self.device = 'cuda' if th.cuda.is_available() else 'cpu'
        self.dist_threhold = dist_threhold

        self.vina_hbond_coeff = nn.Parameter(th.tensor([0.7]))
        self.vina_hydrophobic_coeff = nn.Parameter(th.tensor([0.3]))
        self.vdw_coeff = nn.Parameter(th.tensor([1.0]))
        self.torsion_coeff = nn.Parameter(th.tensor([1.0]))
        self.rotor_coeff = nn.Parameter(th.tensor([1.0]))
        self.intercept = nn.Parameter(th.tensor([0.0]))

        self.decode = nn.Sequential(
            nn.Linear(5, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.num_filter = int(10.0 / 0.5) + 1
        self.filter_center = th.tensor([0.5 * i for i in range(self.num_filter)])
        self.filter_gamma = 10.0
        self.cal_vdw_interaction_A = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.cal_vdw_interaction_B = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )

        self.distance = nn.Linear(3, 1)

        nn.init.uniform_(self.distance.weight, a=0, b=0.5)
        nn.init.uniform_(self.distance.bias, a=0.2, b=0.4)

        print(self.distance.weight)
        print(self.distance.bias)

    def vina_hbond(self, dm, h, vdw_radius1, vdw_radius2, A):

        vdw_radius1_repeat = vdw_radius1.unsqueeze(2) \
            .repeat(1, 1, vdw_radius2.size(1)).unsqueeze(-1)
        vdw_radius2_repeat = vdw_radius2.unsqueeze(1) \
            .repeat(1, vdw_radius1.size(1), 1).unsqueeze(-1)

        B = self.cal_vdw_interaction_B(h) * 0.2

        # dm_0 = (vdw_radius1_repeat + vdw_radius2_repeat + B).squeeze(-1)

        dis_feature = th.cat((vdw_radius1_repeat, vdw_radius2_repeat, B), -1)

        dm_0 = self.distance(dis_feature).squeeze(-1)

        dm = dm - dm_0
        retval = dm * A / -0.7
        retval = retval.clamp(min=0.0, max=1.0)
        coeff = self.vina_hbond_coeff * self.vina_hbond_coeff
        retval = retval * -coeff
        atom_energy = retval.unsqueeze(-1)
        retval = retval.sum(-1).sum(-1).unsqueeze(-1)
        return retval, atom_energy

    def vina_hydrophobic(self, dm, h, vdw_radius1, vdw_radius2, A):

        vdw_radius1_repeat = vdw_radius1.unsqueeze(2) \
            .repeat(1, 1, vdw_radius2.size(1)).unsqueeze(-1)
        vdw_radius2_repeat = vdw_radius2.unsqueeze(1) \
            .repeat(1, vdw_radius1.size(1), 1).unsqueeze(-1)

        B = self.cal_vdw_interaction_B(h) * 0.2

        dis_feature = th.cat((vdw_radius1_repeat, vdw_radius2_repeat, B), -1)

        dm_0 = self.distance(dis_feature).squeeze(-1)

        dm = dm - dm_0

        retval = (-dm + 1.5) * A
        retval = retval.clamp(min=0.0, max=1.0)
        # retval = retval.clamp(min=0.0, max=1.0)*-0.0351
        retval = retval * -self.vina_hydrophobic_coeff \
                 * self.vina_hydrophobic_coeff
        atom_energy = retval.unsqueeze(-1)
        retval = retval.sum(-1).sum(-1).unsqueeze(-1)
        return retval, atom_energy

    def cal_vdw_interaction(self, dm, h, vdw_radius1, vdw_radius2, valid1, valid2):
        vdw_radius1_repeat = vdw_radius1.unsqueeze(2) \
            .repeat(1, 1, vdw_radius2.size(1)).unsqueeze(-1)
        vdw_radius2_repeat = vdw_radius2.unsqueeze(1) \
            .repeat(1, vdw_radius1.size(1), 1).unsqueeze(-1)

        B = self.cal_vdw_interaction_B(h)

        dis_feature = th.cat((vdw_radius1_repeat, vdw_radius2_repeat, B), -1)

        dm_0 = self.distance(dis_feature).squeeze(-1)

        # print(f"distance is {dm_0}")

        valid1_repeat = valid1.unsqueeze(2).repeat(1, 1, valid2.size(1))
        valid2_repeat = valid2.unsqueeze(1).repeat(1, valid1.size(1), 1)

        # dm_0 = vdw_sigma
        dm_0[dm_0 < 0.0001] = 1
        # N = self.cal_vdw_interaction_N(h).squeeze(-1)+5.5
        N = 6.0
        vdw1 = th.pow(dm_0 / dm, 2 * N)
        vdw2 = -2 * th.pow(dm_0 / dm, N)

        A = self.cal_vdw_interaction_A(h).squeeze(-1)

        A = A * (0.0356 - 0.0178)
        A = A + 0.0178

        energy = vdw1 + vdw2
        energy = energy.clamp(max=100)
        energy = energy * valid1_repeat * valid2_repeat
        atom_energy = A * energy
        atom_energy = atom_energy.unsqueeze(-1)
        # energy = energy.unsqueeze(-1)
        energy = atom_energy.sum(1).sum(1)
        
        return energy, B, atom_energy

    def cal_torsion_energy(self, torsion_energy):
        retval = torsion_energy * self.vdw_coeff * self.vdw_coeff
        # retval=torsion_energy*self.torsion_coeff*self.torsion_coeff
        return retval.unsqueeze(-1)

    def cal_distance_matrix(self, p1, p2, dm_min):
        p1_repeat = p1.unsqueeze(2).repeat(1, 1, p2.size(1), 1)
        p2_repeat = p2.unsqueeze(1).repeat(1, p1.size(1), 1, 1)
        dm = th.sqrt(th.pow(p1_repeat - p2_repeat, 2).sum(-1) + 1e-10)
        replace_vec = th.ones_like(dm) * 1e10
        dm = th.where(dm < dm_min, replace_vec, dm)
        return dm

    def mdn_loss_fn(self, pi, sigma, mu, y, eps1=1e-10, eps2=1e-10):
        normal = Normal(mu, sigma)
        loglik = normal.log_prob(y.expand_as(normal.loc))
        prob = (th.log(pi + eps1) + loglik).exp().sum(1)
        loss = -th.log(prob + eps2)
        return loss, prob

    def mdn_score(self, C, B, l_mask, t_mask, N_l, N_t, h_l_pos, h_t_pos):

        self.C_mask = C_mask = l_mask.view(B, N_l, 1) & t_mask.view(B, 1, N_t)
        self.C = C = C[C_mask]
        C = self.MLP(C)

        # Get batch indexes for ligand-target combined features
        C_batch = th.tensor(range(B)).unsqueeze(-1).unsqueeze(-1)
        C_batch = C_batch.repeat(1, N_l, N_t)[C_mask].to(C.device)

        # Outputs
        pi = F.softmax(self.z_pi(C), -1)
        sigma = F.elu(self.z_sigma(C)) + 1.1
        mu = F.elu(self.z_mu(C)) + 1

        dist = self.compute_euclidean_distances_matrix(h_l_pos, h_t_pos.view(B, -1, 3))[C_mask].unsqueeze(1)

        mdn, prob = self.mdn_loss_fn(pi, sigma, mu, dist)

        mdn = mdn[th.where(dist <= 7)[0]]
        mdn = mdn.mean()
        
        prob = prob[th.where(dist <= 5)[0]]
        y = scatter_add(prob, C_batch[th.where(dist <= 5)[0]], dim=0, dim_size=C_batch.unique().size(0)).to(
            th.float32)

        y = y * (-1.36) * 0.03

        
        return mdn, y.unsqueeze(-1)

    def physic_score(self, physic_ligand, physic_target, C, dm, rotor):

        with th.no_grad():
            A_hbond = self.get_A_hbond(physic_ligand[:, :, 3], physic_ligand[:, :, 4], physic_target[:, :, 3],
                                       physic_target[:, :, 4])
            A_metal = self.get_A_hbond(physic_ligand[:, :, 3], physic_ligand[:, :, 5], physic_target[:, :, 3],
                                       physic_target[:, :, 5])
            A_hydro = self.get_outer(physic_ligand[:, :, 2].squeeze(-1), physic_target[:, :, 2].squeeze(-1))

        radius1, radius2 = physic_ligand[:, :, 0].squeeze(-1), physic_target[:, :, 0].squeeze(-1)

        retval = []
        atom_energy = []

        # hbond
        h_energy, h_atom_energy = self.vina_hbond(dm, C, radius1, radius2, A_hbond)
        retval.append(h_energy)
        atom_energy.append(h_atom_energy)

        # metal complex
        metal_energy, metal_atom_energy = self.vina_hbond(dm, C, radius1, radius2, A_metal)
        retval.append(metal_energy)
        atom_energy.append(metal_atom_energy)

        # hydrophobic
        hydro_energy, hydro_atom_energy = self.vina_hydrophobic(dm, C, radius1, radius2, A_hydro)
        retval.append(hydro_energy)
        atom_energy.append(hydro_atom_energy)

        vdw_energy, vdw_radius, vdw_atom_energy = self.cal_vdw_interaction(dm, C, radius1, radius2, physic_ligand[:, :, 1].squeeze(-1),
                                                          physic_target[:, :, 1].squeeze(-1))

        retval.append(vdw_energy)
        atom_energy.append(vdw_atom_energy)

        retval = th.cat(retval, -1)
        atom_energy = th.cat(atom_energy, -1)

        penalty = 1 + self.rotor_coeff * self.rotor_coeff * rotor
        retval = retval / penalty.unsqueeze(-1)

        vdw = retval[:, -1]

        return vdw, retval, vdw_radius, atom_energy


    def forward(self, data_ligand, data_target, train):

        h_l_coord, h_t_coord = data_ligand.pos, data_target.pos
        
        with th.no_grad():
            h_l = self.ligand_model(data_ligand)
            h_t = self.target_model(data_target)

        h_l_pos, _ = to_dense_batch(h_l_coord, h_l.batch, fill_value=10000)
        h_t_pos, _ = to_dense_batch(h_t_coord, h_t.batch, fill_value=10000)
        dm = self.cal_distance_matrix(h_l_pos, h_t_pos, 0.5)

        edge_list = self.get_edge_list(dm, h_l.batch, h_t.batch)

        for module in self.interact_block:
            h_l.x, h_t.x = module(h_l.x, h_t.x, h_l_coord, h_t_coord, edge_list)

        h_l_x, l_mask = to_dense_batch(h_l.x, h_l.batch, fill_value=0)
        h_t_x, t_mask = to_dense_batch(h_t.x, h_t.batch, fill_value=0)

        physic_ligand, _ = to_dense_batch(data_ligand.physic_feats, h_l.batch, fill_value=0)
        physic_target, _ = to_dense_batch(data_target.physic_feats, h_t.batch, fill_value=0)

        # assert h_l_x.size(0) == h_t_x.size(0), 'Encountered unequal batch-sizes'
        (B, N_l, C_out), N_t = h_l_x.size(), h_t_x.size(1)
        self.B = B
        self.N_l = N_l
        self.N_t = N_t

        # Combine and mask
        h_l_x = h_l_x.unsqueeze(-2)
        h_l_x = h_l_x.repeat(1, 1, N_t, 1)  # [B, N_l, N_t, C_out]

        h_t_x = h_t_x.unsqueeze(-3)
        h_t_x = h_t_x.repeat(1, N_l, 1, 1)  # [B, N_l, N_t, C_out]

        C = th.cat((h_l_x, h_t_x), -1)

        vdw, retval, vdw_radius, atom_energy = self.physic_score(physic_ligand, physic_target, C, dm, data_ligand.rotor)
        mdn, mdn_pred = self.mdn_score(C, B, l_mask, t_mask, N_l, N_t, h_l_pos, h_t_pos)

        retval = th.cat((retval, mdn_pred), axis=1)

        # print(retval)

        retval = self.decode(retval)
        # atom_energy = atom_energy.sum(-1)
        # print(atom_energy)

        return vdw, vdw_radius, retval.float(), mdn, mdn_pred, atom_energy

    def get_A_hbond(self, h_acc_indice1, h_donor_indice1, h_acc_indice2, h_donor_indice2):

        A_all = []

        acc1, donor1, acc2, donor2 = \
            h_acc_indice1.squeeze(-1).long(), h_donor_indice1.squeeze(-1).long(), h_acc_indice2.squeeze(
                -1).long(), h_donor_indice2.squeeze(-1).long()

        for i in range(len(h_acc_indice1)):
            A = th.zeros((len(h_acc_indice1[0]), len(h_acc_indice2[0])))

            mask_acc1 = th.cat((acc1[i][:1] >= 0, acc1[i][1:] > 0))
            mask_acc2 = th.cat((acc2[i][:1] >= 0, acc2[i][1:] > 0))
            mask_donor1 = th.cat((donor1[i][:1] >= 0, donor1[i][1:] > 0))
            mask_donor2 = th.cat((donor2[i][:1] >= 0, donor2[i][1:] > 0))

            acc_indice1 = th.masked_select(acc1[i], mask_acc1).unsqueeze(1)
            donor_indice2 = th.masked_select(donor2[i], mask_donor2).unsqueeze(0)
            A[acc_indice1, donor_indice2] = 1

            acc_indice2 = th.masked_select(acc2[i], mask_acc2).unsqueeze(1)
            donor_indice1 = th.masked_select(donor1[i], mask_donor1).unsqueeze(0)
            A[donor_indice1, acc_indice2] = 1

            A_all.append(A)

        A_all = th.stack(A_all).to(h_acc_indice1.device)

        return A_all

    def get_outer(self, h_acc_indice1, h_acc_indice2):

        A_all = []

        for i in range(len(h_acc_indice1)):
            A = th.outer(h_acc_indice1[i], h_acc_indice2[i])

            A_all.append(A)

        A_all = th.stack(A_all).to(h_acc_indice1.device)

        return A_all

    def get_edge_list(self, dm, batch1, batch2):

        adj12 = dm.clone().detach()

        adj12[adj12 > 5] = 0
        adj12[adj12 > 1e-3] = 1
        adj12[adj12 < 1e-3] = 0

        h_l_count = th.bincount(batch1)
        h_p_count = th.bincount(batch2)
        num_l = 0
        num_p = 0
        row_all = []
        col_all = []

        for i in range(len(adj12)):
            rows, cols = th.where(adj12[i] == 1)
            row_all.append(rows + num_l)
            col_all.append(cols + num_p)
            num_l += h_l_count[i]
            num_p += h_p_count[i]

        row_all = th.cat(row_all, dim=0)
        col_all = th.cat(col_all, dim=0)

        edges = [row_all, col_all]

        return edges

    def compute_euclidean_distances_matrix(self, X, Y):
        # Based on: https://medium.com/@souravdey/l2-distance-matrix-vectorization-trick-26aa3247ac6c
        # (X-Y)^2 = X^2 + Y^2 -2XY
        X = X.double()
        Y = Y.double()

        dists = -2 * th.bmm(X, Y.permute(0, 2, 1)) + th.sum(Y ** 2, axis=-1).unsqueeze(1) + th.sum(X ** 2,
                                                                                                   axis=-1).unsqueeze(
            -1)
        return th.nan_to_num((dists ** 0.5), 10000)


