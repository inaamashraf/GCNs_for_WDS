from typing import List, Optional, Union
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import ( Dropout, Sequential, SELU)
from torch_scatter import scatter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size

""" 
    Using partial code from torch_geometric.nn.conv.GENconv
    https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gen_conv.html#GENConv
    """

class MLP(Sequential):
    def __init__(self, dims: List[int], bias: bool = True, dropout: float = 0., activ=SELU()):
        m = []
        for i in range(1, len(dims)):
            m.append(Linear(dims[i - 1], dims[i], bias=bias))

            if i < len(dims) - 1:                
                m.append(activ)
                m.append(Dropout(dropout))

        super().__init__(*m)

class GENConvolution(MessagePassing):
    r"""
    Args:
        in_dim (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_dim (int): Size of each output sample.
        edge_dim (int): Size of edge features.
        aggr (str, optional): The aggregation scheme to use (:obj:`"softmax"`,
            :obj:`"softmax_sg"`, :obj:`"power"`, :obj:`"add"`, :obj:`"mean"`,
            :obj:`max`). (default: :obj:`"softmax"`)        
        num_layers (int, optional): The number of MLP layers.
            (default: :obj:`2`)
        eps (float, optional): The epsilon value of the message construction
            function. (default: :obj:`1e-7`)
        bias (bool, optional): If set to :obj:`False`, will not use bias. 
            (default: :obj:`True`)
        dropout (float, optional): Percentage of neurons to be dropped in MLP.
            (default: :obj:`0.`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GenMessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{in}), (|\mathcal{V_t}|, F_{t})`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge attributes :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """
    def __init__(self, in_dim: int, out_dim: int, edge_dim: int,
                 aggr: str = 'add', num_layers: int = 2, eps: float = 1e-7, 
                 bias: bool = True, dropout: float = 0., **kwargs):

        kwargs.setdefault('aggr', None)
        super().__init__(**kwargs)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim
        self.aggr = aggr
        self.eps = eps
        
        assert aggr in ['add', 'mean', 'max']

        dims = [self.in_dim]
        for i in range(num_layers - 1):
            dims.append(2 * in_dim)
        dims.append(self.out_dim)
        self.mlp = MLP(dims, bias=bias, dropout=dropout)

        """ Added a linear layer to manage dimensionality """
        self.res = Linear(in_dim + edge_dim, in_dim, bias=bias)

    def reset_parameters(self):
        if self.msg_norm is not None:
            self.msg_norm.reset_parameters()
        if self.t and isinstance(self.t, Tensor):
            self.t.data.fill_(self.initial_t)
        if self.p and isinstance(self.p, Tensor):
            self.p.data.fill_(self.initial_p)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None, 
                residual: bool = True, mlp: bool = True) -> Tensor:
        """"""
                       
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)        
        x_in = x[0]
        
        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        sndr_node_attr = torch.gather(x_in, 0, edge_index[0:1,:].repeat(x_in.shape[1], 1).T)
        rcvr_node_attr = torch.gather(x_in, 0, edge_index[1:2,:].repeat(x_in.shape[1], 1).T)
        edge_attr = edge_attr + (sndr_node_attr - rcvr_node_attr).abs()
        
        latent = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size) 
        
        """ Added a linear layer to manage dimensionality """
        if mlp:
            latent = self.res(latent)
        else:
            latent = torch.tanh(self.res(latent))

        del sndr_node_attr, rcvr_node_attr
        
        if residual:
            latent = latent + x[1]
        
        del x, edge_index, edge_attr 
        if mlp:
            latent = self.mlp(latent)  
        return latent       

    def message(self, x_j: Tensor, edge_attr: OptTensor) -> Tensor:
        """ Concatenating edge features instead of adding those to node features """
        msg = x_j if edge_attr is None else torch.cat((x_j, edge_attr), dim=1)
        del x_j, edge_attr
        return F.selu(msg) + self.eps

    def aggregate(self, inputs: Tensor, index: Tensor,
                  dim_size: Optional[int] = None) -> Tensor:

        return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size,
                           reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_dim}, '
                f'{self.out_dim}, aggr={self.aggr})')
