import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_geometric.data import Data
import pickle as pkl
import os
from copy import deepcopy
import random
import torch.optim as optim
import time
import glob
from torch_geometric.utils import dense_to_sparse
from sklearn.decomposition import PCA
from PIL import Image
device = 'cpu'


class MultiModalGraphTransformer(nn.Module):
    def __init__(self, in_channels=2, hidden_dim=16, num_classes=3, num_layers=1, heads=4, device=device):
        """
        Args:
            in_channels (int): Number of input features (e.g., 2: temperature and greyscale).
            hidden_dim (int): Dimension of hidden embeddings.
            num_classes (int): Number of output classes (e.g., 3 for quality levels).
            num_layers (int): Number of graph transformer layers.
            heads (int): Number of attention heads.
        """
        super(MultiModalGraphTransformer, self).__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.out_channels = hidden_dim // heads

        # Linear projection for edge_attr
        self.lin_edge = nn.Linear(1, heads * self.out_channels, bias=False)


        #Feature embedding: Transform input features -> hidden_dim
        self.feature_embedding = nn.Linear(in_channels, hidden_dim)

        # Graph Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=0.1)
            for _ in range(num_layers)
        ])

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x, edge_index, edge_attr, batch, fit=False):
        """
        Args:
            x (Tensor): Node features (shape: [num_nodes, in_channels]).
            edge_index (LongTensor): Graph connectivity (shape: [2, num_edges]).
            edge_attr (Tensor): Edge weights (shape: [num_edges]).
            batch (LongTensor): Batch vector for pooling (shape: [num_nodes]).
        Returns:
            out (Tensor): Class logits for each graph in the batch.
        """
        # Feature embedding
        x = self.feature_embedding(x)  # Shape: [num_nodes, hidden_dim]

        # Apply graph transformer layers with weighted edges
        if edge_attr is not None:
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
        for layer in self.transformer_layers:
            x = layer(x, edge_index, edge_attr=edge_attr)  # Pass edge_attr to TransformerConv
            x = F.relu(x)  # Non-linearity

        # Global pooling to aggregate node features into graph-level features
        x = global_mean_pool(x, batch)  # Shape: [num_graphs, hidden_dim]

        # Classification
        out = self.classifier(x)  # Shape: [num_graphs, num_classes]
        return out


"""class AutoencoderGreyscale(nn.Module):
    def __init__(self, output, device='cpu'):
        super(AutoencoderGreyscale, self).__init__()

        #Encoder
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3,3), padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=(3,3), padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        #Decoder
        self.conv3 = nn.Conv2d(8, 4, kernel_size=(3, 3), padding=1)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv4 = nn.Conv2d(4, 8, kernel_size=(3, 3), padding=1)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv5 = nn.Conv2d(8, 1, kernel_size=(3, 3), padding=1)


    def forward(self, x, fit=False, lr=0.001):

        if fit:
            x = x.detach()
            x.requires_grad=True

        #Encoder
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        # Decoder
        x = F.relu(self.conv3(x))
        x = self.up1(x)
        x = F.relu(self.conv4(x))
        x = self.up2(x)
        x = torch.sigmoid(self.conv5(x))
        y= 0

        return y, x"""

class AutoencoderGreyscale_pretrained(nn.Module):
    def __init__(self):
        super(AutoencoderGreyscale_pretrained, self).__init__()

        #Encoder
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3,3), padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=(3,3), padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        #Decoder
        self.conv3 = nn.Conv2d(8, 4, kernel_size=(3, 3), padding=1)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv4 = nn.Conv2d(4, 8, kernel_size=(3, 3), padding=1)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv5 = nn.Conv2d(8, 1, kernel_size=(3, 3), padding=1)


    def forward(self, x, fit=False, lr=0.001):

        if fit:
            x = x.detach()
            x.requires_grad=True

        #Encoder
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        # Decoder
        y = F.relu(self.conv3(x))
        y = self.up1(y)
        y = F.relu(self.conv4(y))
        y = self.up2(y)
        y = torch.sigmoid(self.conv5(y))


        return x,y


class AutoencoderGreyscale(nn.Module):
    def __init__(self, latent_dim=(25, 25), device=device):
        super(AutoencoderGreyscale, self).__init__()
        self.device = device
        self.latent_dim = latent_dim

        # Encoder
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=(3, 3), padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Reduce to 1 channel
        self.reduce_channels = nn.Conv2d(8, 1, kernel_size=(1, 1))
        self.poolem = nn.AdaptiveMaxPool2d(output_size=self.latent_dim)

        # Decoder
        self.conv3 = nn.Conv2d(1, 8, kernel_size=(3, 3), padding=1)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv4 = nn.Conv2d(8, 16, kernel_size=(3, 3), padding=1)
        self.up2 = nn.Upsample(size=(100, 100), mode='nearest')
        self.conv5 = nn.Conv2d(16, 1, kernel_size=(3, 3), padding=1)

        """# Encoder
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=(3, 3), padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        #self.reduce_channels = nn.Conv2d(8, 1, kernel_size=(1, 1))

        # Decoder
        self.conv3 = nn.Conv2d(8, 4, kernel_size=(3, 3), padding=1)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv4 = nn.Conv2d(4, 8, kernel_size=(3, 3), padding=1)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv5 = nn.Conv2d(8, 1, kernel_size=(3, 3), padding=1)"""

    def set_latent_dim(self, new_latent_dim):

        self.latent_dim = new_latent_dim
        self.poolem = nn.AdaptiveMaxPool2d(output_size=self.latent_dim)

    def principal_components(self, encoded_data):
        samples = encoded_data.detach().numpy().shape
        encoded_data_reshaped = encoded_data.reshape((samples[0], samples[1] * samples[2] * samples[3]))
        transposed_encoded_data = encoded_data_reshaped.T
        pca = PCA(n_components=1)
        pca.fit(transposed_encoded_data)
        return pca.components_

    def forward(self, x, fit=False):
        if fit:
            x = x.detach()
            x.requires_grad = True

        # Encoder
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        # Reduce channels to 1
        x = F.relu(self.reduce_channels(x))
        # Explicitly reshape latent space
        latent_space = self.poolem(x)  # Apply adaptive pooling


        # Decoder
        y = F.relu(self.conv3(latent_space))
        y = self.up1(y)
        y = F.relu(self.conv4(y))
        y = self.up2(y)
        y = torch.sigmoid(self.conv5(y))

        """latent_space = 0
        # Encoder
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        #x = F.relu(self.reduce_channels(x))

        # Decoder
        x = F.relu(self.conv3(x))
        x = self.up1(x)
        x = F.relu(self.conv4(x))
        x = self.up2(x)
        x = torch.sigmoid(self.conv5(x))"""

        return latent_space, y

class ConnectivityModel(nn.Module):
    """
    Model for determining the connectivity for the heat flow along the (directed) edges in the graph ('\varphi')
    """
    
    def __init__(self, hidden_dims=[256,],loss_weights=[1.,1.,.1,.1],device=device):
        super().__init__()
        
        self.loss_weights = torch.tensor(loss_weights, dtype=torch.float32,device=device) # weights for the regularization loss functions



        self.temp_regulariser = nn.Parameter(torch.tensor(1e-3,dtype=torch.float32, device=device)) # multiplier for the input temperature values
        self.layer_1 = nn.Linear(12,hidden_dims[0],device=device)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dims[i],hidden_dims[i+1],device=device) for i in range(len(hidden_dims)-1)])
        self.final_layer = nn.Linear(hidden_dims[-1],1,device=device)
        torch.nn.init.normal_(self.final_layer.weight, mean=0.0, std=.01)
        
    def forward(self, t1, t2, c1, c2, dens1, dens2, dist, fit=False, eps=1e-9):

        if fit: # enable gradients for the inputs in order to calculate the regularization losses
            t1 = t1.detach()
            t1.requires_grad=True
            t2 = t2.detach()
            t2.requires_grad=True
            dens1 = dens1.detach()
            dens1.requires_grad=True
            dens2 = dens2.detach()
            dens2.requires_grad=True
            dist = dist.detach()
            dist.requires_grad=True

        # normalization of the input temperatures
        t1 = torch.tanh(self.temp_regulariser * t1)
        t2 = torch.tanh(self.temp_regulariser * t2)

        # concatenate the argument tensors
        args = torch.cat([t1.unsqueeze(-1), t2.unsqueeze(-1), c1, c2, dist.unsqueeze(-1), (t2-t1).unsqueeze(-1), dens1.unsqueeze(-1), dens2.unsqueeze(-1)], dim=-1)

        # evaluate the model
        x = torch.tanh(self.layer_1(args))
        for layer in self.hidden_layers:
            x = torch.tanh(layer(x))
        x = self.final_layer(x)
        x = F.softplus(x)

        if fit: # calculate the regularization losses
            grad = torch.autograd.grad([x[i] for i in range(len(x))],[t1,t2,dens1,dens2,dist],retain_graph=True)
            dist_loss = ((grad[-1]/(x.squeeze(-1)+eps))-(-2/(dist+eps))).square().mean()
            temp_loss = grad[0].square().mean() + grad[1].square().mean()
            dens_loss = (grad[2]/(x.squeeze(-1)+eps)).square().mean() + (grad[3]/(x.squeeze()+eps)).square().mean()
            # ((grad[2]/(x.squeeze(-1)+eps))-(1/(dens1+eps))).square().mean() + ((grad[3]/(x.squeeze()+eps))-(-1/(dens2+eps))).square().mean()
            symmetry_loss = (x-self(t2, t1, c2, c1, dens1, dens2, dist)).square().mean()
            conn_loss = torch.dot(torch.stack([dist_loss,temp_loss,dens_loss,symmetry_loss]),self.loss_weights)
            return x, conn_loss
        return x


class DissipationModel(nn.Module):
    """
    Model for estimating the heat dissipation at each vertex ('\psi')
    """
    
    def __init__(self, hidden_dims=[256,], boundary_temp=124.9, device=device):
        super().__init__()
        
        self.temp_regulariser = nn.Parameter(torch.tensor(1e-2,dtype=torch.float32, device=device)) # multiplier for the input temperature values
        
        self.layer_1 = nn.Linear(5, hidden_dims[0],device=device)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dims[i],hidden_dims[i+1],device=device) for i in range(len(hidden_dims)-1)])
        self.final_layer = nn.Linear(hidden_dims[-1], 1,device=device)
        torch.nn.init.normal_(self.final_layer.weight, mean=0.0, std=.01)
        
        self.coefs = nn.Parameter(-torch.ones(3,dtype=torch.float32, device=device)) # learnable parameters in the regularization loss 
        
    def forward(self,temperature, classes, density, fit=False):
        
        if fit:#enable gradients for the inputs in order to calculate the regularization loss
            temperature = temperature.detach()
            temperature.requires_grad=True
            density = density.detach()
            density.requires_grad=True
            
        temperature = torch.tanh(self.temp_regulariser * temperature) # normalization of the input temperatures
        
        # concatenate the argument tensors
        args = torch.cat([temperature.unsqueeze(-1), classes, density.unsqueeze(-1)], dim=-1)
        
        # evaluate the model
        x = torch.tanh(self.layer_1(args))
        for layer in self.hidden_layers:
            x = torch.tanh(layer(x))
        x = self.final_layer(x)
        
        if fit:#calculate the regularization losses
            grad = torch.autograd.grad([x[i] for i in range(len(x))],[temperature, density],retain_graph=True)
            bottom_ix = torch.where(classes[:, 0] == 1, True, False)
            top_ix = torch.where(classes[:, 1] == 1, True, False)
            side_ix = torch.where(classes[:, 2] == 1, True, False)
            interior_ix = torch.where(classes.sum(dim=1) == 0, True, False)
            temp_loss = (grad[0] - (-torch.exp(self.coefs[0]))).where(bottom_ix,torch.zeros_like(grad[0])).square().mean() + \
                        (grad[0] - (-torch.exp(self.coefs[1]))).where(top_ix,torch.zeros_like(grad[0])).square().mean() + \
                        (grad[0] - (-torch.exp(self.coefs[2]))).where(side_ix,torch.zeros_like(grad[0])).square().mean() + \
                        grad[0].where(interior_ix, torch.zeros_like(grad[0])).square().mean()
            dens_loss = grad[1].square().mean()
            return x, temp_loss + dens_loss
        return x

    
class LaserModel(nn.Module):
    """
    Model for the temperature change caused by the laser
    """
    def __init__(self,device=device):
        super().__init__()
        self.intensity = nn.Parameter(50*torch.ones(1))
        self.decay = nn.Parameter(.4*torch.ones(1))
        
    def forward(self, distances):
        return self.intensity * torch.exp(-(self.decay*distances).square())
        """A = 0.09
        r = 0.00014
        p = 250
        v = 200e-3
        A = 0.09
        D = 140e-6
        R = D / 2
        P = 250
        v = 200e-3
        x_max = 1e-3
        y_max = 0.23e-3
        z_max = 0.1e-3
        t_max = 0.002
        x_test = torch.linspace(0, x_max, 10)
        y_test = torch.linspace(0, y_max, 10)
        t_test = torch.linspace(0, t_max, 10)
        X_test, Y_test, T_test = torch.meshgrid(x_test, y_test, t_test)
        X = torch.vstack((torch.ravel(X_test), torch.ravel(Y_test), torch.ravel(T_test))).T
        r_sq = (X[:, 0:1] - x_max / 4 - X[:, 2:3] * v) ** 2 + (X[:, 1:2] - y_max / 2) ** 2
        k = (11.82 + 1.06e-2 * X[:,1:2])
        k_eff = 0.6 * k
        heat_Q = 2 * A * P / (torch.pi * R ** 2) * torch.exp(-2 * r_sq / R ** 2) / k_eff
        return heat_Q"""


class CNModel(nn.Module):
    def __init__(self, k=4, boundary_value=124.9, device=device):
        super().__init__()
        self.boundary_value = boundary_value
        self.k = k
        
        # initialize the submodels
        self.conn_model = ConnectivityModel(device=device)
        self.diss_model = DissipationModel(device=device)
        self.laser_model = LaserModel(device=device)
        
        
    def forward(self, distance_adj, densities, vertex_class, temperature, dt, laser_dist=None, fit=False, eps=1e-9):
        
        if laser_dist is None: # if no laser is present, assign a very large distance to the laser position for each vertex
            laser_dist = 1e9 * torch.ones_like(temperature)
        
        # prepare arguments for the submodels
        c1 = vertex_class[distance_adj.indices()[0]]
        c2 = vertex_class[distance_adj.indices()[1]]
        
        dens1 = densities[distance_adj.indices()[0]]
        dens2 = densities[distance_adj.indices()[1]]
        
        dists = distance_adj.values()
        
        t1 = torch.index_select(temperature,0,distance_adj.indices()[0]) - self.boundary_value
        t2 = torch.index_select(temperature,0,distance_adj.indices()[1]) - self.boundary_value

        grad_X = (t1-t2)/dists
        grad_magnitude = torch.sqrt(grad_X**2)
        
        # determine the edge connectivity for the current state 
        if fit:
            connectivity, conn_loss = self.conn_model(t1,t2,c1,c2,dens1,dens2,dists,fit=True)
        else:
            connectivity = self.conn_model(t1, t2, c1, c2, dens1, dens2, dists)
        
        # create the graph laplacian from the edge connectivities
        connectivity = connectivity.squeeze(-1)
        conn_matrix = torch.sparse_coo_tensor(distance_adj.indices(),connectivity,distance_adj.shape)
        conn_matrix = conn_matrix.coalesce()
        degree = torch.sparse.sum(conn_matrix,dim=1)
        degree_matrix = torch.sparse_coo_tensor(torch.stack([degree.indices()[0],degree.indices()[0]],dim=0), degree.values(), conn_matrix.shape)
        laplacian = conn_matrix - degree_matrix
        
        # evaluate the dissipation model
        if fit:
            diss_vector, diss_loss = self.diss_model(temperature.view(-1),vertex_class,densities,fit=True)
        else:
            diss_vector = self.diss_model(temperature.view(-1),vertex_class,densities)
        
        # iterate over k and calculate the Crank-Nicolson time-steps
        diff_temp = temperature.unsqueeze(-1) + .5 * dt * diss_vector
        idty = torch.eye(*laplacian.shape).to_sparse()
        for _ in range(self.k):
            diff_temp = torch.linalg.solve((idty-.5*dt/self.k*laplacian).to_dense(), torch.sparse.mm((idty+.5*dt/self.k*laplacian), diff_temp).to_dense())
        
        new_temp = (diff_temp + .5 * dt * diss_vector).squeeze(-1)
        
        
        # predict and add the temperature change caused by the laser
        laser_heat = self.laser_model(laser_dist).squeeze(-1)

        ##########################################################################
        """x = laser_heat.unsqueeze(0).unsqueeze(0)  # → shape (1, 1, 1000)
        laser_heat = F.interpolate(x, size=new_temp.size(), mode='linear', align_corners=False)
        laser_heat = laser_heat.squeeze()  # → shape (50,)
        new_temp = new_temp + dt * laser_heat"""
        ##########################################################################
        new_temp = new_temp + dt * laser_heat
        
        if fit: # return the predicted value, together with the values for the regularizing loss functions
            return new_temp, conn_loss, diss_loss, dt * diss_vector, dt * laser_heat
        return new_temp
    
    
    def develop(self,X,initial_state=None):
        """
        use the model to update the internal heat state of the part, using an initial state and taking surface data into account
        """
        
        if initial_state is not None:
            state = initial_state.detach().clone()
        else:
            state = self.boundary_value * torch.ones((X[0][0].shape[0],), dtype=torch.float32)
        
        for distances, densities, boundary, surface_temp, time_delta in X:
            state[surface_temp.indices()] = 0.
            state = state + surface_temp.to_dense()
            pred = self(distances, densities, boundary, state, time_delta)
            state = pred.detach().view(-1)
        return state
    
    
    def save(self,path,compiled=False, override=False):
        if os.path.isfile(path) and not override:
            raise ValueError('file already exists!')
        if compiled:
            torch.save(deepcopy(self),path)
        else:
            model_state = {
                'state_dt': deepcopy(self.state_dict()),
                'boundary_value': self.boundary_value
            }
            torch.save(model_state,path)
    
    @staticmethod
    def load(path, compiled=False):
        if compiled:
            loaded_model = torch.load(path)
            return loaded_model
        model = CNModel()
        model_state = torch.load(path)
        model.load_state_dict(model_state['state_dt'])
        model.boundary_value = model_state['boundary_value']
        return model



##################################################### Mulitmodal CN ########################################
class MultiModalCNModel(nn.Module):
    def __init__(self, k=4, boundary_value=124.9, output_dim=(9,1), device=device):
        super().__init__()
        self.boundary_value = boundary_value
        self.k = k

        # initialize the submodels
        self.conn_model = ConnectivityModel(device=device)
        self.diss_model = DissipationModel(device=device)
        self.laser_model = LaserModel(device=device)
        self.autoencoder_model = AutoencoderGreyscale(output_dim, device=device)
        self.graph_transformer = MultiModalGraphTransformer(device=device)

    def forward(self, distance_adj, densities, vertex_class, temperature, dt, img, y, laser_dist=None, fit=False, qual=True, eps=1e-9):

        if laser_dist is None:  #if no laser is present, assign a very large distance to the laser position for each vertex
            laser_dist = 1e9 * torch.ones_like(temperature)

        # prepare arguments for the submodels
        c1 = vertex_class[distance_adj.indices()[0]]
        c2 = vertex_class[distance_adj.indices()[1]]

        dens1 = densities[distance_adj.indices()[0]]
        dens2 = densities[distance_adj.indices()[1]]

        dists = distance_adj.values()

        t1 = torch.index_select(temperature, 0, distance_adj.indices()[0]) - self.boundary_value
        t2 = torch.index_select(temperature, 0, distance_adj.indices()[1]) - self.boundary_value

        # determine the edge connectivity for the current state
        if fit:
            connectivity, conn_loss = self.conn_model(t1, t2, c1, c2, dens1, dens2, dists, fit=True)
        else:
            connectivity = self.conn_model(t1, t2, c1, c2, dens1, dens2, dists)

        # create the graph laplacian from the edge connectivities
        connectivity = connectivity.squeeze(-1)
        conn_matrix = torch.sparse_coo_tensor(distance_adj.indices(), connectivity, distance_adj.shape)
        conn_matrix = conn_matrix.coalesce()
        degree = torch.sparse.sum(conn_matrix, dim=1)
        degree_matrix = torch.sparse_coo_tensor(torch.stack([degree.indices()[0], degree.indices()[0]], dim=0),
                                                degree.values(), conn_matrix.shape)
        laplacian = degree_matrix - conn_matrix
        # evaluate the dissipation model
        if fit:
            diss_vector, diss_loss = self.diss_model(temperature.view(-1), vertex_class, densities, fit=True)
        else:
            diss_vector = self.diss_model(temperature.view(-1), vertex_class, densities)



        # evaluate the autoencoder model
        ########## pretrained version ###################
        #autoencoder = AutoencoderGreyscale_pretrained()
        #autoencoder.load_state_dict(torch.load("best_autoencoder.pth"))
        #latent, pred_grey = autoencoder(img)
        output_dim = y.values().size(dim=0)

        # Hauptkomponenten berechnen
        def principal_components(encoded_data, n_comp):
            encoded_data = encoded_data.detach().numpy()
            samples = encoded_data.shape
            encoded_data_reshaped = encoded_data.reshape((samples[0], samples[1] * samples[2] * samples[3]))
            transposed_encoded_data = encoded_data_reshaped.T
            pca = PCA(n_components=2)
            pca.fit(transposed_encoded_data)
            return pca.components_

        #emb = principal_components(latent, output_dim)

        #########################################################
        #output_dim = y.values().size(dim=0)
        self.autoencoder_model.set_latent_dim((output_dim, 1))
        if fit:
            emb, pred_grey = self.autoencoder_model(img, fit=True)
        else:
            emb, pred_grey = self.autoencoder_model(img)
        # iterate over k and calculate the Crank-Nicolson time-steps
        diff_temp = temperature.unsqueeze(-1) + .5 * dt * diss_vector
        idty = torch.eye(*laplacian.shape).to_sparse()
        for _ in range(self.k):
            diff_temp = torch.linalg.solve((idty - .5 * dt / self.k * laplacian).to_dense(),
                                           torch.sparse.mm((idty + .5 * dt / self.k * laplacian), diff_temp).to_dense())

        new_temp = (diff_temp + .5 * dt * diss_vector).squeeze(-1)

        # predict and add the temperature change caused by the laser
        laser_heat = self.laser_model(laser_dist).squeeze(-1)
        new_temp = new_temp + dt * laser_heat

        grey_init = torch.ones(new_temp.size(dim=0))*0.5

        multifeat = torch.stack((new_temp, grey_init))
        multifeat = multifeat.T
        multifeat[y.indices(), 1] = emb[0,0,:,0]

        if conn_matrix.is_sparse:
            conn_matrix = conn_matrix.to_dense()
        edge_index, edge_attr = dense_to_sparse(conn_matrix)

        # Assuming multifeat, edge_index, and edge_attr are defined
        # Ensure edge_index is of type int64
        edge_index = edge_index.long()

        # If edge_attr exists, ensure its type matches expected float precision
        if edge_attr is not None:
            edge_attr = edge_attr.float()



        assert edge_index.size(1) == edge_attr.size(
            0), "Mismatch: edge_index and edge_attr must have the same number of edges"

        batch = torch.zeros(new_temp.size(dim=0), dtype=torch.long)
        edge_attr = edge_attr.view(-1, 1)

        # Temporal spatial graph transformer
        if fit:
            out = self.graph_transformer(multifeat, edge_index, edge_attr, batch, fit=True)
        else:
            out = self.graph_transformer(multifeat, edge_index, edge_attr, batch)


        if fit:  # return the predicted value, together with the values for the regularizing loss functions
            return new_temp, conn_loss, diss_loss, dt * diss_vector, dt * laser_heat, pred_grey, out
        return new_temp, out, pred_grey

    def develop(self, X, initial_state=None):
        """
        use the model to update the internal heat state of the part, using an initial state and taking surface data into account
        """

        if initial_state is not None:
            state = initial_state.detach().clone()
        else:
            state = self.boundary_value * torch.ones((X[0][0].shape[0],), dtype=torch.float32)

        for distances, densities, boundary, surface_temp, time_delta in X:
            state[surface_temp.indices()] = 0.
            state = state + surface_temp.to_dense()
            pred = self(distances, densities, boundary, state, time_delta)
            state = pred.detach().view(-1)
        return state

    def save(self, path, compiled=False, override=False):
        if os.path.isfile(path) and not override:
            raise ValueError('file already exists!')
        if compiled:
            torch.save(deepcopy(self), path)
            torch.save(self.autoencoder_model.state_dict(), 'models/Autoencoder.pt')
        else:
            model_state = {
                'state_dt': deepcopy(self.state_dict()),
                'boundary_value': self.boundary_value
            }
            torch.save(model_state, path)

            def save(self, path, compiled=False, override=False):
                if os.path.isfile(path) and not override:
                    raise ValueError('file already exists!')
                if compiled:
                    torch.save(deepcopy(self), path)
                else:
                    model_state = {
                        'state_dt': deepcopy(self.state_dict()),
                        'boundary_value': self.boundary_value
                    }
                    torch.save(model_state, path)

    @staticmethod
    def load(path, compiled=False):
        if compiled:
            loaded_model = torch.load(path)
            return loaded_model
        model = MultiModalCNModel()
        model_state = torch.load(path)
        model.load_state_dict(model_state['state_dt'])
        model.boundary_value = model_state['boundary_value']
        return model

class MultiModalCNModelbatch(nn.Module):
    def __init__(self, k=4, boundary_value=124.9, output_dim=(9, 1), device=device):
        super().__init__()
        self.boundary_value = boundary_value
        self.k = k

        # initialize the submodels
        self.conn_model = ConnectivityModel(device=device)
        self.diss_model = DissipationModel(device=device)
        self.laser_model = LaserModel(device=device)
        #self.autoencoder_model = AutoencoderGreyscale(output_dim, device=device)
        #self.graph_transformer = MultiModalGraphTransformer(device=device)

    def forward(self, distance_adj, densities, vertex_class, temperature, dt, laser_dist=None, fit=False):

        if laser_dist is None:  # if no laser is present, assign a very large distance to the laser position for each vertex
            laser_dist = 1e9 * torch.ones_like(temperature)

        # prepare arguments for the submodels
        c1 = vertex_class[distance_adj.indices()[0]]
        c2 = vertex_class[distance_adj.indices()[1]]

        dens1 = densities[distance_adj.indices()[0]]
        dens2 = densities[distance_adj.indices()[1]]

        dists = distance_adj.values()

        t1 = torch.index_select(temperature, 0, distance_adj.indices()[0]) - self.boundary_value
        t2 = torch.index_select(temperature, 0, distance_adj.indices()[1]) - self.boundary_value

        # determine the edge connectivity for the current state
        if fit:
            connectivity, conn_loss = self.conn_model(t1, t2, c1, c2, dens1, dens2, dists, fit=True)
        else:
            connectivity = self.conn_model(t1, t2, c1, c2, dens1, dens2, dists)

        # create the graph laplacian from the edge connectivities
        connectivity = connectivity.squeeze(-1)
        conn_matrix = torch.sparse_coo_tensor(distance_adj.indices(), connectivity, distance_adj.shape)
        conn_matrix = conn_matrix.coalesce()
        degree = torch.sparse.sum(conn_matrix, dim=1)
        degree_matrix = torch.sparse_coo_tensor(torch.stack([degree.indices()[0], degree.indices()[0]], dim=0),
                                                degree.values(), conn_matrix.shape)
        laplacian = degree_matrix - conn_matrix
        # evaluate the dissipation model
        if fit:
            diss_vector, diss_loss = self.diss_model(temperature.view(-1), vertex_class, densities, fit=True)
        else:
            diss_vector = self.diss_model(temperature.view(-1), vertex_class, densities)

        # evaluate the autoencoder model
        #output_dim = y.values().size(dim=0)
        #autoencoder = AutoencoderGreyscale_pretrained()
        """self.autoencoder_model.set_latent_dim((output_dim, 1))
        if fit:
            emb, pred_grey = self.autoencoder_model(img, fit=True)
        else:
            emb, pred_grey = self.autoencoder_model(img)"""

        ############################################################################################
        #iterate over k and calculate the Crank-Nicolson time-steps
        diff_temp = temperature.unsqueeze(-1) + .5 * dt * diss_vector
        idty = torch.eye(*laplacian.shape).to_sparse()
        for _ in range(self.k):
            diff_temp = torch.linalg.solve((idty - .5 * dt / self.k * laplacian).to_dense(),
                                           torch.sparse.mm((idty + .5 * dt / self.k * laplacian),
                                                           diff_temp).to_dense())

        new_temp = (diff_temp + .5 * dt * diss_vector).squeeze(-1)

        # predict and add the temperature change caused by the laser
        laser_heat = self.laser_model(laser_dist).squeeze(-1)
        new_temp = new_temp + dt * laser_heat

        grey_init = torch.ones(new_temp.size(dim=0)) * 0.5

        multifeat = torch.stack((new_temp, grey_init))
        multifeat = multifeat.T
        #multifeat[y.indices(), 1] = emb[0, 0, :, 0]

        if conn_matrix.is_sparse:
            conn_matrix = conn_matrix.to_dense()
        edge_index, edge_attr = dense_to_sparse(conn_matrix)

        # Assuming multifeat, edge_index, and edge_attr are defined
        # Ensure edge_index is of type int64
        edge_index = edge_index.long()

        # If edge_attr exists, ensure its type matches expected float precision
        if edge_attr is not None:
            edge_attr = edge_attr.float()

        assert edge_index.size(1) == edge_attr.size(
            0), "Mismatch: edge_index and edge_attr must have the same number of edges"

        batch = torch.zeros(new_temp.size(dim=0), dtype=torch.long)
        edge_attr = edge_attr.view(-1, 1)

        # Temporal spatial graph transformer
        """if fit:
            out = self.graph_transformer(multifeat, edge_index, edge_attr, batch, fit=True)
        else:
            out = self.graph_transformer(multifeat, edge_index, edge_attr, batch)
        out = torch.tensor([[0.46874, 10.4749, -9.038]])"""
        if fit:  # return the predicted value, together with the values for the regularizing loss functions
            return new_temp, conn_loss, diss_loss, dt * diss_vector, dt * laser_heat, edge_attr, edge_index
        return new_temp

    def develop(self, X, initial_state=None):
        """
        use the model to update the internal heat state of the part, using an initial state and taking surface data into account
        """

        if initial_state is not None:
            state = initial_state.detach().clone()
        else:
            state = self.boundary_value * torch.ones((X[0][0].shape[0],), dtype=torch.float32)

        for distances, densities, boundary, surface_temp, time_delta in X:
            state[surface_temp.indices()] = 0.
            state = state + surface_temp.to_dense()
            pred = self(distances, densities, boundary, state, time_delta)
            state = pred.detach().view(-1)
        return state

    def save(self, path, compiled=False, override=False):
        if os.path.isfile(path) and not override:
            raise ValueError('file already exists!')
        if compiled:
            torch.save(deepcopy(self), path)
            #torch.save(self.autoencoder_model.state_dict(), 'models/Autoencoder.pt')
        else:
            model_state = {
                'state_dt': deepcopy(self.state_dict()),
                'boundary_value': self.boundary_value
            }
            torch.save(model_state, path)

            def save(self, path, compiled=False, override=False):
                if os.path.isfile(path) and not override:
                    raise ValueError('file already exists!')
                if compiled:
                    torch.save(deepcopy(self), path)
                else:
                    model_state = {
                        'state_dt': deepcopy(self.state_dict()),
                        'boundary_value': self.boundary_value
                    }
                    torch.save(model_state, path)

    @staticmethod
    def load(path, compiled=False):
        if compiled:
            loaded_model = torch.load(path)
            return loaded_model
        model = MultiModalCNModel()
        model_state = torch.load(path)
        model.load_state_dict(model_state['state_dt'])
        model.boundary_value = model_state['boundary_value']
        return model

#######################################################################################################################
    
inv_ = np.vectorize(lambda x: 1/x if not x == 0. else 0.)



def scale_invariant_density(space,return_avg_dist=False):
    """
    calculate the scale-invariant-density, as defined in: https://doi.org/10.48550/arXiv.2110.01286
    with adaptation for 3d data
    """
    
    ret_val = None
    dim = space.shape[-1]
    if dim != 2 and dim != 3:
        print(space.shape)
        raise NotImplementedError()
    pairings = np.tile(space,(space.shape[0],1,1)) - np.tile(space,(space.shape[0],1,1)).transpose([1,0,2])
    dens = np.sum(np.square(pairings),axis=-1)
    if dim == 2:
        ret_val = np.sum(inv_(np.sqrt(dens)), axis=1)
    else:
        ret_val =  np.sum(inv_(dens), axis=1)
    
    if return_avg_dist:
        return ret_val, np.linalg.norm(pairings.reshape(-1,dim),axis=-1).mean()
    return ret_val

def load_data_layer(layer,device=device,obj="pyramid_7", layers="571_to_1079", min_layer=571, max_layer=1063, vertex_multipliers=None, print_info=False):
    """
    load the previously created graphs and the extracted surface data of the given layer
    """
    
    with open(f'{obj}/layer_{min_layer+layer}.pkl','rb') as f:
        layer_data = pkl.load(f)
    with open(f'{obj}_adjacencies/layers_{layers}/layer_{layer}.pkl','rb') as f:
        graph_data = pkl.load(f)
    with open(f'{obj}_graphs/layers_{layers}/layer_{layer}.pkl','rb') as f:
        simplex_data = pkl.load(f)

    if print_info:
        print(layer_data[0][0])
        print(layer_data[0][-1])
        print(len(layer_data[0]))        
    timestamps, temperatures, laser_position = layer_data
    times_ms = torch.tensor([(ts.asm8.astype('int')/1e6) for ts in timestamps], dtype=torch.float32,device=device)
    vertices,distance_t, bottom_boundary,top_boundary,side_boundary = graph_data
    distance_t = distance_t.coalesce()
    simplices = simplex_data[1]
    
    top_layer_indices = top_boundary.to_sparse().indices()[0]
    top_layer_vertices = vertices[top_layer_indices][:, :2].numpy()
    
    if vertex_multipliers is not None:
        top_layer_vertices[:, 0] *= 1/vertex_multipliers[0]
        top_layer_vertices[:, 1] *= 1/vertex_multipliers[1]
        top_layer_vertices = np.round(top_layer_vertices)

    top_layer_vertices = top_layer_vertices.astype(int)

    top_layer_x, top_layer_y = list(zip(*top_layer_vertices))
    top_layer_x, top_layer_y = torch.tensor(top_layer_x), torch.tensor(top_layer_y)
    
    temp_values = torch.tensor(np.array([temp_image[top_layer_x, top_layer_y] for temp_image in temperatures]), dtype=torch.float32,device=device)
    temp_vecs = [torch.sparse_coo_tensor(top_layer_indices.unsqueeze(0), temp_vals, (len(vertices),), dtype=torch.float32,device=device).coalesce() for temp_vals in temp_values]
    
    boundary_masks = torch.stack([bottom_boundary,top_boundary,side_boundary],dim=-1)
    
    densities = torch.tensor(scale_invariant_density(vertices), dtype=torch.float32)
    densities = densities/densities.mean()
    
    laser_distance = []
    for laser_pos in laser_position:
        if laser_pos is None:
            laser_distance.append(1e9 * torch.ones(len(vertices)))
        else:
            top_layer_z = vertices[:,2].max()
            laser_distance.append(torch.tensor([torch.linalg.norm(vertex[:2]-laser_pos) if vertex[2] == top_layer_z else 1e9 for vertex in vertices], dtype=torch.float32, device=device))
    
    return vertices.to(device), distance_t.to(device), densities.to(device), boundary_masks.to(device), times_ms, temp_vecs, laser_distance, simplices


############################### load data layer - multimodal ####################################
def load_data_layer_multimodal(layer, device=device, obj="pyramid_7", layers="571_to_1079", min_layer=571, max_layer=958,
                    vertex_multipliers=None, print_info=False):
    """
    load the previously created graphs and the extracted surface data of the given layer
    """

    """image_path = f'greyscale_data/{obj}/layer_{min_layer + layer}.jpg'

    # Load the single image
    image = Image.open(image_path)

    # Convert the image to a numpy array and normalize it
    image_array = np.array(image).astype('float32') / 255.0

    # Reshape the image to add a channel dimension
    image_array = np.reshape(image_array, (1, 100, 100, 1))  # 1 for batch size if needed"""

    filelist = glob.glob(f'greyscale_data/{obj}/layer_{min_layer + layer}.jpg')
    y = np.array([np.array(Image.open(fname)) for fname in filelist])
    y = y.astype('float32') / 255.
    y = np.reshape(y, (1, 1, 100, 100))

    # Convert to a PyTorch tensor
    y_tensor = torch.tensor(y, device=device)

    with open(f'{obj}/layer_{min_layer + layer}.pkl', 'rb') as f:
        layer_data = pkl.load(f)
    with open(f'{obj}_adjacencies/layers_{layers}/layer_{layer}.pkl', 'rb') as f:
        graph_data = pkl.load(f)
    with open(f'{obj}_graphs/layers_{layers}/layer_{layer}.pkl', 'rb') as f:
        simplex_data = pkl.load(f)

    if print_info:
        print(layer_data[0][0])
        print(layer_data[0][-1])
        print(len(layer_data[0]))
    timestamps, temperatures, laser_position = layer_data
    times_ms = torch.tensor([(ts.asm8.astype('int') / 1e6) for ts in timestamps], dtype=torch.float32, device=device)
    vertices, distance_t, bottom_boundary, top_boundary, side_boundary = graph_data
    distance_t = distance_t.coalesce()
    simplices = simplex_data[1]

    top_layer_indices = top_boundary.to_sparse().indices()[0]
    top_layer_vertices = vertices[top_layer_indices][:, :2].numpy()

    if vertex_multipliers is not None:
        top_layer_vertices[:, 0] *= 1 / vertex_multipliers[0]
        top_layer_vertices[:, 1] *= 1 / vertex_multipliers[1]
        top_layer_vertices = np.round(top_layer_vertices)

    top_layer_vertices = top_layer_vertices.astype(int)

    top_layer_x, top_layer_y = list(zip(*top_layer_vertices))
    top_layer_x, top_layer_y = torch.tensor(top_layer_x), torch.tensor(top_layer_y)

    temp_values = torch.tensor(np.array([temp_image[top_layer_x, top_layer_y] for temp_image in temperatures]),
                               dtype=torch.float32, device=device)
    temp_vecs = [
        torch.sparse_coo_tensor(top_layer_indices.unsqueeze(0), temp_vals, (len(vertices),), dtype=torch.float32,
                                device=device).coalesce() for temp_vals in temp_values]

    boundary_masks = torch.stack([bottom_boundary, top_boundary, side_boundary], dim=-1)

    densities = torch.tensor(scale_invariant_density(vertices), dtype=torch.float32)
    densities = densities / densities.mean()

    laser_distance = []
    for laser_pos in laser_position:
        if laser_pos is None:
            laser_distance.append(1e9 * torch.ones(len(vertices)))
        else:
            top_layer_z = vertices[:, 2].max()
            laser_distance.append(torch.tensor(
                [torch.linalg.norm(vertex[:2] - laser_pos) if vertex[2] == top_layer_z else 1e9 for vertex in vertices],
                dtype=torch.float32, device=device))

    return vertices.to(device), distance_t.to(device), densities.to(device), boundary_masks.to(
        device), times_ms, temp_vecs, laser_distance, simplices, y_tensor

#################################################################################################


def load_surface_temperatures(layer,obj='pyramid_7', start=571):
    with open(f'{obj}/layer_{start+layer}.pkl','rb') as f:
        layer_data = pkl.load(f)
    return layer_data[1]


def random_transform(image_tensor):
    """
    Apply random flipping and rotation (0°, 90°, 180°, 270°) to an image tensor.
    Assumes input shape is (batch_size, channels, height, width)
    """
    # Random flip
    if random.choice([True, False]):
        image_tensor = image_tensor.flip(dims=[2])  # Vertical flip (height axis)
    if random.choice([True, False]):
        image_tensor = image_tensor.flip(dims=[3])  # Horizontal flip (width axis)

    # Random rotation (0, 90, 180, 270 degrees)
    k = random.randint(0, 3)  # Number of 90° rotations
    if k == 1:  # 90°
        image_tensor = image_tensor.transpose(2, 3).flip(dims=[2])
    elif k == 2:  # 180°
        image_tensor = image_tensor.flip(dims=[2, 3])
    elif k == 3:  # 270°
        image_tensor = image_tensor.transpose(2, 3).flip(dims=[3])

    return image_tensor

