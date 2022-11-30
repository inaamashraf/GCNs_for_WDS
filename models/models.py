import torch
import torch.nn.functional as F
from torch_geometric.nn import ChebConv
from models.layers import GENConvolution
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
    Graph convolution based model using multiple GCN layers (with multiple hops).
"""
class m_GCN(torch.nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, latent_dim=96, n_aggr=45, n_hops=1, bias=False, num_layers=2, dropout=0., batch_size=48):
        super(m_GCN, self).__init__()
        self.n_aggr = n_aggr
        self.n_hops = n_hops
        self.batch_size = batch_size
        self.latent = latent_dim
        self.out_dim = out_dim
        self.node_in = torch.nn.Linear(in_dim, latent_dim, bias=bias)        
        self.node_out = torch.nn.Linear(latent_dim, out_dim, bias=bias)
        self.edge = torch.nn.Linear(edge_dim, latent_dim, bias=bias)        
            
        self.gcn_aggrs = torch.nn.ModuleList()
        for _ in range(n_aggr):
            gcn = GENConvolution(latent_dim, latent_dim, latent_dim, aggr="add", bias=bias, num_layers=num_layers, dropout=dropout)
            self.gcn_aggrs.append(gcn)
        
    def forward(self, data):
        data = data.to(device)
        x, y, edge_index, edge_attr  = data.x, data.y, data.edge_index, data.edge_attr        
      
        """ Embedding for edge features. """
        edge_attr = self.edge(edge_attr)

        """ Embedding for node features. """
        Z = self.node_in(x)

        """ 
            Mutiple GCN layers.
        """
        for gcn in self.gcn_aggrs:
            """
                Multiple Hops.
            """
            for _ in range(self.n_hops - 1):
                Z = torch.selu(gcn(Z, edge_index, edge_attr, mlp=False))
            Z = torch.selu(gcn(Z, edge_index, edge_attr, mlp=True))
                
        """ Reconstructing node features through a final dense layer. """
        y_predict = self.node_out(Z)
        
        del data
        
        return y, y_predict

    def loss(self, y, y_predict):
        """ L1 Loss. """       
        loss = torch.nn.L1Loss(reduction='mean')             
        _loss = loss(y, y_predict)         
        del y, y_predict 
        return _loss


"""
    ChebNet Model from Hajgato et al. 
    https://github.com/BME-SmartLab/GraphConvWat
"""
class ChebNet(torch.nn.Module):
    def __init__(self, in_dim, out_dim, latent_dim, K):
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(in_dim, latent_dim, K=K)
        self.conv2 = ChebConv(latent_dim, 60, K=120)
        self.conv3 = ChebConv(60, 30, K=20)
        self.conv4 = ChebConv(30, out_dim, K=1, bias=False)

    def forward(self, data):
        data = data.to(device)
        x, y, edge_index, edge_weight  = data.x, data.y, data.edge_index, data.edge_weight        
        del data        
        x = F.silu(self.conv1(x, edge_index, edge_weight))
        x = F.silu(self.conv2(x, edge_index, edge_weight))
        x = F.silu(self.conv3(x, edge_index, edge_weight))
        x = self.conv4(x, edge_index, edge_weight)
        return y, torch.sigmoid(x)        
        
    def loss(self, y, y_predict):
        _loss = torch.nn.MSELoss()
        loss = _loss(y, y_predict)
        del y, y_predict
        return loss

