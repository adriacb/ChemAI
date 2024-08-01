# https://projects.volkamerlab.org/teachopencadd/talktorials/T036_e3_equivariant_gnn.html
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.aggr import SumAggregation
from torch_geometric.data import Data
from torch import Tensor, LongTensor
from torch_scatter import scatter
from typing import Optional
# since the task is to predict properties that are directly
# linked to an Euclidean space, we can use a simple graph neural network

# the term E(n)-Equivariant Graph Neural Network means that the network
# is equivariant to the group of Euclidean isometries, which means that
# the network should be able to predict the same properties regardless
# of the orientation of the molecule in space.

# - the Euclidean group E(n), which consists of all distance-preserving transformations, and

# - the special Euclidean group SE(n), which consists only of translations and rotations.

# If we consider a model that makes predictions about objects which are coupled to the Euclidean space 
# X' = theta(X,Z) belonging R^m*n (e.g. future atom positions in a dynamical system), we can define 
# E(n)-equivariance as: theta(g(X),g(Z)) = g(theta(X,Z)) for all g belonging to E(n).
# This is saying that the output of theta is transformed in the same way as the input when the input is transformed.

class EquivariantMPLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        act: nn.Module,
    ) -> None:
        super().__init__()
        self.act = act
        self.residual_proj = nn.Linear(in_channels, hidden_channels, bias=False)

        # Messages will consist of two (source and target) node embeddings and a scalar distance
        message_input_size = 2 * in_channels + 1

        # equation (3) "phi_l" NN
        self.message_mlp = nn.Sequential(
            nn.Linear(message_input_size, hidden_channels),
            act,
        )
        # equation (4) "psi_l" NN
        self.node_update_mlp = nn.Sequential(
            nn.Linear(in_channels + hidden_channels, hidden_channels),
            act,
        )

    def node_message_function(
        self,
        source_node_embed: Tensor,  # h_i
        target_node_embed: Tensor,  # h_j
        node_dist: Tensor,          # d_ij
    ) -> Tensor:
        """
        Compute the messages "m_ij" between pairs of nodes.
        
        m_ij^{l} = phi_l(h_i, h_j, d_ij) for l = 0, 1, ..., L - 1 (3)
        
        Parameters
        ----------
        source_node_embed : Tensor
            The source node embeddings.
        target_node_embed : Tensor
            The target node embeddings.
        node_dist : Tensor
            The relative squared distances between the nodes.
            
        Returns
        -------
        Tensor
            The messages "m_ij" between the nodes.
        """
        message_repr = torch.cat((source_node_embed, target_node_embed, node_dist), dim=-1)
        return self.message_mlp(message_repr)

    def compute_distances(self, node_pos: Tensor, edge_index: LongTensor) -> Tensor:
        """
        Compute the relative squared distances between all pairs of nodes
        in the graph.
        
        Parameters
        ----------
        node_pos : Tensor
            The node positions in 3D space.
        edge_index : LongTensor
            The edge index tensor.
        
        Returns
        -------
        Tensor
            The relative squared distances between all pairs of nodes.
        """
        row, col = edge_index
        xi, xj = node_pos[row], node_pos[col]
        # relative squared distance
        # implements equation (2)                                            d_ij = ||X_i - X_j||^2
        rsdist = (xi - xj).pow(2).sum(1, keepdim=True)
        return rsdist

    def forward(
        self,
        node_embed: Tensor,
        node_pos: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        row, col = edge_index
        dist = self.compute_distances(node_pos, edge_index)

        # compute messages "m_ij" from  equation (3)
        node_messages = self.node_message_function(node_embed[row], node_embed[col], dist)

        # message sum aggregation in equation (4)
        aggr_node_messages = scatter(node_messages, col, dim=0, reduce="sum")

        # compute new node embeddings "h_i^{l+1}"
        # (implements rest of equation (4))
        new_node_embed = self.residual_proj(node_embed) + self.node_update_mlp(
            torch.cat((node_embed, aggr_node_messages), dim=-1)
        )

        return new_node_embed
    
class EquivariantGNN(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        final_embedding_size: Optional[int] = None,
        target_size: int = 1,
        num_mp_layers: int = 2,
    ) -> None:
        super().__init__()
        if final_embedding_size is None:
            final_embedding_size = hidden_channels

        # non-linear activation func.
        # usually configurable, here we just use Relu for simplicity
        self.act = nn.ReLU()

        # equation (1) "psi_0"
        self.f_initial_embed = nn.Embedding(100, hidden_channels)

        # create stack of message passing layers
        self.message_passing_layers = nn.ModuleList()
        channels = [hidden_channels] * (num_mp_layers) + [final_embedding_size]
        for d_in, d_out in zip(channels[:-1], channels[1:]):
            layer = EquivariantMPLayer(d_in, d_out, self.act)
            self.message_passing_layers.append(layer)

        # modules required for readout of a graph-level
        # representation and graph-level property prediction
        self.aggregation = SumAggregation()
        self.f_predict = nn.Sequential(
            nn.Linear(final_embedding_size, final_embedding_size),
            self.act,
            nn.Linear(final_embedding_size, target_size),
        )

    def encode(self, data: Data) -> Tensor:
        # theory, equation (1)
        node_embed = self.f_initial_embed(data.z)
        # message passing
        # theory, equation (3-4)
        for mp_layer in self.message_passing_layers:
            # NOTE here we use the complete edge index defined by the transform earlier on
            # to implement the sum over $j \neq i$ in equation (4)
            node_embed = mp_layer(node_embed, data.pos, data.complete_edge_index)
        return node_embed

    def _predict(self, node_embed, batch_index) -> Tensor:
        aggr = self.aggregation(node_embed, batch_index)
        return self.f_predict(aggr)

    def forward(self, data: Data) -> Tensor:
        node_embed = self.encode(data)
        pred = self._predict(node_embed, data.batch)
        return pred