import time
import math
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from GenScore.model.transform_AK.transform import SubgraphsTransform
from GenScore.model.transform_AK.config import cfg, update_cfg

from GenScore.model.transform_AK.element import MLP, DiscreteEncoder, Identity, VNUpdate
from torch_scatter import scatter_mean, scatter
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class subgraph(nn.Module):
    def __init__(self, model, dim_gnn):
        super(subgraph, self).__init__()

        self.embs = (0, 1, 2)

        self.dropout = 0.

        self.hop_embedding = nn.Linear(144, dim_gnn, bias=False)

        self.GAT_conv = model

        self.hop_embedder = nn.Embedding(20, 16)

        self.subgraph_transform = MLP(128, 128, nlayer=1, with_final_activation=True)
        self.context_transform = MLP(128, 128, nlayer=1, with_final_activation=True)
        self.gate_mapper_subgraph = nn.Sequential(nn.Linear(16, 128), nn.Sigmoid())
        self.gate_mapper_context = nn.Sequential(nn.Linear(16, 128), nn.Sigmoid())
        self.gate_mapper_centroid = nn.Sequential(nn.Linear(16, 128), nn.Sigmoid())
        self.pooling = 'mean'
        self.embs_combine_mode = 'concat'
        self.out_encoder = MLP(128 if self.embs_combine_mode == 'add' else 128 * len(self.embs), 128, nlayer=1,
                               with_final_activation=False, bias=True, with_norm=True)


    def forward(self, data, sample, model_type=None):

        combined_subgraphs_x = data.x[sample['subgraphs_nodes_mapper']]
        combined_subgraphs_coords = data.pos[sample['subgraphs_nodes_mapper']]
        data.pos = combined_subgraphs_coords
        combined_subgraphs_edge_attr = data.edge_attr[sample['subgraphs_edges_mapper']]
        combined_subgraphs_edge_index = sample['combined_subgraphs']
        data.edge_index = combined_subgraphs_edge_index
        combined_subgraphs_batch = sample['subgraphs_batch']
        hop_emb = self.hop_embedder(sample['hop_indicator'] + 1)
        combined_subgraphs_x = torch.cat([combined_subgraphs_x, hop_emb], dim=-1)
        combined_subgraphs_x = self.hop_embedding(combined_subgraphs_x)

        if model_type == 'dynamic':
            for gt_layer in self.GAT_conv:
                combined_subgraphs_x = gt_layer(combined_subgraphs_x, combined_subgraphs_edge_index, combined_subgraphs_edge_attr)

        else:
            # Apply a given number of intermediate geometric attention layers to the node and edge features given
            for gt_layer in self.GAT_conv[:-1]:
                combined_subgraphs_x, combined_subgraphs_edge_attr = gt_layer(data, combined_subgraphs_x, combined_subgraphs_edge_attr)

            # Apply final layer to update node representations by merging current node and edge representations
            combined_subgraphs_x = self.GAT_conv[-1](data, combined_subgraphs_x, combined_subgraphs_edge_attr)

        centroid_x = combined_subgraphs_x[(sample['subgraphs_nodes_mapper'] == combined_subgraphs_batch)]
        subgraph_x = self.subgraph_transform(
            F.dropout(combined_subgraphs_x, self.dropout, training=self.training)) if len(
            self.embs) > 1 else combined_subgraphs_x
        context_x = self.context_transform(
            F.dropout(combined_subgraphs_x, self.dropout, training=self.training)) if len(
            self.embs) > 1 else combined_subgraphs_x

        centroid_x = centroid_x * self.gate_mapper_centroid(
            hop_emb[(sample['subgraphs_nodes_mapper'] == combined_subgraphs_batch)])
        subgraph_x = subgraph_x * self.gate_mapper_subgraph(hop_emb)
        context_x = context_x * self.gate_mapper_context(hop_emb)
        subgraph_x = scatter(subgraph_x, combined_subgraphs_batch, dim=0, reduce=self.pooling)
        context_x = scatter(context_x, sample['subgraphs_nodes_mapper'], dim=0, reduce=self.pooling)

        x = [centroid_x, subgraph_x, context_x]
        x = [x[i] for i in self.embs]
        if self.embs_combine_mode == 'add':
            x = sum(x)
        else:
            x = torch.cat(x, dim=-1)
            # last part is only essential for embs_combine_mode = 'concat', can be ignored when overfitting
            x = self.out_encoder(F.dropout(x, self.dropout, training=self.training))

        return x
