import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from DiffusionModel import *
from ImplicitModel import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle as pkl
from tqdm import tqdm
import os
from copy import deepcopy
import time
import datetime as dt
from sklearn.random_projection import GaussianRandomProjection
from scipy.spatial import Delaunay
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def transfer_state(vertices_source, state_source, vertices_target, space_dim=(100,100,100000)):
    """
    transfer the temperatures in the source graph to the target vertices at the same spatial location 
    (for passing from the graph for one layer to the graph for the next layer)
    """
    
    device = state_source.device
    indices = torch.stack([vertices_source[:,0].type(torch.int32),
                           vertices_source[:,1].type(torch.int32),
                           (1000*vertices_source[:,2]).type(torch.int32)])

    space_temperatures = torch.sparse_coo_tensor(indices,state_source.to_dense(),space_dim,device=device)
    state_target = torch.stack([space_temperatures[int(x),int(y),int(1000*z)] for x,y,z in vertices_target])
    return state_target


def calculate_energy(distribution, density, classes):
    """
    calculate the potential energy of the heat state (i.e. the second moment of the temperature over the domain)
    """
    
    boundary_order = classes.sum(dim=1)
    weights = (1/density)*torch.pow(2.,-boundary_order)
    weights = weights/weights.sum()
    energy = torch.dot(distribution.square(), weights) - torch.dot(distribution, weights).square()
    return energy

def calculate_heat(distribution, density, classes, dissipation=None):
    """
    calculate the total thermal energy 
    """
    
    boundary_order = classes.sum(dim=1)
    weights = (1/density)*torch.pow(2.,-boundary_order)
    heat = torch.dot(distribution,weights)
    if dissipation is None:
        return heat
    dissipation_heat = torch.dot(dissipation,weights)
    return heat, dissipation_heat
    


def fit_multi_model(model,
              depths_iters=[(2, 250), (5, 100),(10, 50)],
                #(10, 50), (25, 20), (50, 10), (100, 5), (200, 3)
              # pairs of training depth and number of training iterations for the depth
              initial_state=None,
              lambd=1,
              alpha_conn=1.,
              alpha_diss=1.,
              alpha_energy=1.,
              alpha_heat=1.,
              alpha_max_min=1.,
              t_max=1e3,
              lr=1e-5,
              betas=(.5, .999),
              save_best=False,
              save_path='Test1.pt',
              obj='pyramid_9',
              min_layer=571,
              max_layer=958,
              vertex_multipliers=None,
              set_plot_limits=True,
              whole_layer=False,
              device=device):
    
    
    n_layers = depths_iters[-1][0] #total number of layers in the final part


    optim = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
    loss_fn = nn.MSELoss()
    #loss_fn_feat = nn.MSELoss(reduction='none')
    bce_loss_fn = nn.CrossEntropyLoss()

    full_iters = depths_iters[-1][1]
    scheduler = torch.optim.lr_scheduler.LinearLR(optim, start_factor=1., end_factor=1. / full_iters,
                                               total_iters=full_iters)

    history = []
    energy_loss = []
    start_time = int(time.time())

    save_depth = False

    # increase the number of layers used for training, as later layers are sensitive to the hidden state obtained from the model predictions, and
    # thus require a relatively well-trained model in order to produce reasonable predictions
    for dix, (depth, depth_iterations) in enumerate(depths_iters):
        print('depth:', depth)

        if dix == (len(depths_iters) - 1):
            save_depth = True
            best_loss = np.inf

        if not whole_layer and depth >= 5:
            ditr = range(depth_iterations)
            pbar = True
        else:
            ditr = tqdm(range(depth_iterations))
            pbar = False

        for depth_iter in ditr:

            # initialize values for accumulating the losses for the current training depth
            depth_avg_loss = 0.
            depth_fwd_loss = 0.

            if pbar:
                itr = tqdm(range(depth))
            else:
                itr = range(depth)

            for layer in itr:

                data = load_data_layer_multimodal(layer, obj=obj, layers=f'{min_layer}_to_{max_layer}', min_layer=min_layer,
                                       max_layer=max_layer, vertex_multipliers=vertex_multipliers, device=device)

                # initialize values for accumulating losses for the current layer
                layer_loss = 0.
                layer_const_loss = 0.
                layer_conn_loss = 0.
                layer_diss_loss = 0.
                layer_energy = 0.
                layer_avg_steps = 0.
                layer_loss_greyscale = 0.
                layer_quality = 0.
                correct = 0.
                total = 0.

                if layer == 0:
                    if initial_state is not None:
                        layer_init_state = initial_state
                    else:
                        layer_init_state = model.boundary_value * torch.ones((data[0].shape[0],), dtype=torch.float32,
                                                                             device=device)

                # determine the initial heat state for the layer by transfering the final state of the previous layer, taking into account the
                # diffusion in the time passed during recoating
                else:
                    time_delta = (data[4][0] - prev_last_time) / 1000
                    state, output, grey = model(prev_distances, prev_densities, prev_boundary, state, time_delta, transformed_tensor, y)
                    state = state.detach().view(-1)
                    layer_init_state = transfer_state(prev_vertices, state, data[0])

                prev_vertices = data[0]
                prev_distances = data[1]
                prev_densities = data[2]
                prev_boundary = data[3]
                prev_last_time = data[4][-1]

                state = layer_init_state

                # sample the evaluation time-steps (either random or n-step)

                # jumps = np.random.poisson(lambd,len(data[5])) + 1
                # eval_ixs = np.cumsum(jumps)
                # eval_ixs = eval_ixs[np.where(eval_ixs<len(data[5]))].tolist()
                # if eval_ixs[0] > 0:
                #     eval_ixs = [0,] + eval_ixs

                if whole_layer:
                    eval_ixs = [0, len(data[5]) - 1]
                else:
                    eval_ixs = np.arange(0, len(data[5]), lambd).tolist()
                    if eval_ixs[-1] < len(data[5]) - 1:
                        eval_ixs.append(len(data[5]) - 1)

                X = [(data[1], data[2], data[3], data[5][eval_ixs[i]],
                      (data[4][eval_ixs[i + 1]] - data[4][eval_ixs[i]]) / 1000, data[6][eval_ixs[i + 1]]) for i in
                     range(len(eval_ixs) - 1)]
                Y = [data[5][eval_ixs[i]] for i in range(1, len(eval_ixs))]
                count = 0
                for (distances, densities, boundary, surface_temp, time_delta, laser_dist), y in zip(X, Y):

                    optim.zero_grad()

                    # assign the surface temperature values
                    state[surface_temp.indices()[0]] = 0.
                    state = state + surface_temp.to_dense()

                    # Data Augmentation ##########################
                    transformed_tensor = random_transform(data[8])
                    #output_dim = y.values().size(dim=0)
                    #evaluate the model and loss function
                    pred, conn_loss, diss_loss, diss_vec, laser_heat, pred_grey, quality_pred = model(distances.to(device), densities.to(device),
                                                                             boundary.to(device), state.to(device),
                                                                             time_delta.to(device), transformed_tensor.to(device), y,
                                                                             laser_dist.to(device), fit=True)
                    pred_loss = loss_fn(pred[y.indices()].view(-1, 1), y.values().view(-1, 1))
                    greyscale_loss = loss_fn(pred_grey, transformed_tensor)

                    if 0 <= layer <= 40:
                        lab = 0
                    elif 41 <= layer <= 110:
                        lab = 1
                    else:
                        lab = 2
                    #high_qual = torch.tensor([1., 0., 0.])
                    #high_qual = high_qual.unsqueeze(0)


                    #print(f"quality_pred shape: {quality_pred.shape}")  # Expected: [1, 3]
                    #print(f"high_qual shape: {high_qual.shape}")  # Expected: [1, 3]
                    target = torch.tensor([lab]).to(quality_pred.device)
                    qual_loss = bce_loss_fn(quality_pred, target)
                    predictions = torch.argmax(quality_pred, dim=1)
                    #print(qual_loss.detach().numpy())
                    total += target.size(0)
                    correct += (predictions == target).sum().item()
                    #print(quality_pred.detach().numpy())

                    # calculate the regularizing loss functions
                    pred = pred - laser_heat
                    neighbor_temp = (torch.sign(distances).to_dense() * pred.view(-1).to_dense())
                    max_principle_violation = (torch.relu(pred.view(-1) - neighbor_temp.max(dim=1).values)).where(
                        torch.logical_and(pred > state, boundary.sum(dim=1) == 0),
                        torch.zeros_like(pred)).square().mean()
                    min_principle_violation = (torch.relu(neighbor_temp.min(dim=1).values - pred.view(-1))).where(
                        torch.logical_and(pred > state, boundary.sum(dim=1) == 0),
                        torch.zeros_like(pred)).square().mean()
                    energy_violation = torch.relu(
                        calculate_energy(pred, densities, boundary)  # energy of prediction
                        - calculate_energy(state, densities, boundary)  # energy of previous state
                    )
                    heat, heat_diss = calculate_heat(pred, densities, boundary, dissipation=diss_vec.squeeze())
                    total_heat_diff = ((heat - heat_diss) / state.sum() - 1).square()




                    # assemble the total loss
                    loss = pred_loss + \
                           alpha_conn * conn_loss + \
                           alpha_diss * diss_loss + \
                           alpha_energy * energy_violation + \
                           alpha_heat * total_heat_diff + \
                           alpha_max_min * (max_principle_violation + min_principle_violation) +\
                           greyscale_loss +\
                           qual_loss


                    # optimization step
                    loss.backward()
                    optim.step()

                    # log the loss values
                    push_fwd_loss = loss_fn(state[y.indices()].view(-1, 1), y.values().view(-1, 1))
                    layer_loss += pred_loss.item() / (len(eval_ixs) - 1)
                    layer_const_loss += push_fwd_loss.item() / (len(eval_ixs) - 1)
                    layer_conn_loss += conn_loss.item() / (len(eval_ixs) - 1)
                    layer_diss_loss += diss_loss.item() / (len(eval_ixs) - 1)
                    layer_energy += (
                                                energy_violation + total_heat_diff + max_principle_violation + min_principle_violation).item() / (
                                                len(eval_ixs) - 1)

                    layer_loss_greyscale += greyscale_loss.item() / (len(eval_ixs) - 1)
                    layer_quality += qual_loss.item() / (len(eval_ixs) - 1)




                    # use the predicted state for the next time-step
                    state = pred.detach().clip(0, t_max).view(-1)

                layer_avg_steps += len(eval_ixs) - 1
                accuracy = correct / total
                if pbar:
                    if layer_const_loss != 0.:
                        itr.set_postfix({'l_T': layer_loss,
                                         'rel': (layer_const_loss - layer_loss) / layer_const_loss,
                                         'l_reg': (layer_conn_loss + layer_diss_loss),
                                         'l_E': layer_energy,
                                         't_min': state.min().item(),
                                         't_max': state.max().item(),
                                        'l_grey': layer_loss_greyscale,
                                        'qual': layer_quality,
                                        'accuracy': accuracy})


                # energy_loss.append(layer_energy)
                depth_avg_loss += layer_loss
                depth_fwd_loss += layer_const_loss

            if not depth_fwd_loss == 0:
                if pbar:
                    print(depth_avg_loss / depth, (depth_fwd_loss - depth_avg_loss) / depth_fwd_loss)
                else:
                    ditr.set_postfix(
                        {'l_T': depth_avg_loss / depth, 'rel': (depth_fwd_loss - depth_avg_loss) / depth_fwd_loss})
            if save_depth:
                scheduler.step()
                if depth_avg_loss < best_loss:
                    best_loss = depth_avg_loss
                    print('saving model state.')
                    model.save(save_path, compiled=True, override=True)

        # plot_state(data[0].detach().numpy(), state.detach().numpy(),data[7], set_limits=set_plot_limits, save_path=f'plots/depth_{depth}_plot{dix}.svg')
        print('parameters:')
        print(model.diss_model.coefs)
        print(model.laser_model.intensity.item(), ', ', model.laser_model.decay.item(), sep='')

    return history


def fit_multi_model_batch(model,
              depths_iters=[(2, 250), (5, 100),(10, 50), (25, 20), (50, 10), (100, 5), (200, 3)],
              # pairs of training depth and number of training iterations for the depth
              initial_state=None,
              lambd=1,
              alpha_conn=1.,
              alpha_diss=1.,
              alpha_energy=1.,
              alpha_heat=1.,
              alpha_max_min=1.,
              t_max=1e3,
              lr=1e-5,
              betas=(.5, .999),
              save_best=False,
              save_path='Test1.pt',
              obj='pyramid_9',
              min_layer=571,
              max_layer=958,
              vertex_multipliers=None,
              set_plot_limits=True,
              whole_layer=False,
              device=device):
    n_layers = depths_iters[-1][0] #total number of layers in the final part

    graph_transformer = MultiModalGraphTransformer()
    autoencoder = AutoencoderGreyscale_pretrained()
    autoencoder.load_state_dict(torch.load("best_autoencoder.pth"))

    optim = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
    optim_MPGT = torch.optim.Adam(graph_transformer.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    #loss_fn_feat = nn.MSELoss(reduction='none')
    bce_loss_fn = nn.CrossEntropyLoss()

    full_iters = depths_iters[-1][1]
    scheduler = torch.optim.lr_scheduler.LinearLR(optim, start_factor=1., end_factor=1. / full_iters,
                                               total_iters=full_iters)

    history = []
    energy_loss = []
    start_time = int(time.time())

    save_depth = False
    testcount = 0
    # increase the number of layers used for training, as later layers are sensitive to the hidden state obtained from the model predictions, and
    # thus require a relatively well-trained model in order to produce reasonable predictions
    for dix, (depth, depth_iterations) in enumerate(depths_iters):
        print('depth:', depth)

        if dix == (len(depths_iters) - 1):
            save_depth = True
            best_loss = np.inf

        if not whole_layer and depth >= 5:
            ditr = range(depth_iterations)
            pbar = True
        else:
            ditr = tqdm(range(depth_iterations))
            pbar = False

        for depth_iter in ditr:

            # initialize values for accumulating the losses for the current training depth
            depth_avg_loss = 0.
            depth_fwd_loss = 0.

            if pbar:
                itr = tqdm(range(depth))
            else:
                itr = range(depth)

            for layer in itr:
                testcount = testcount + 1
                data = load_data_layer_multimodal(layer, obj=obj, layers=f'{min_layer}_to_{max_layer}', min_layer=min_layer,
                                       max_layer=max_layer, vertex_multipliers=vertex_multipliers, device=device)

                # initialize values for accumulating losses for the current layer
                layer_loss = 0.
                layer_const_loss = 0.
                layer_conn_loss = 0.
                layer_diss_loss = 0.
                layer_energy = 0.
                layer_avg_steps = 0.
                layer_loss_greyscale = 0.
                layer_quality = 0.
                correct = 0.
                total = 0.
                graph_pred_temp = []
                grayscale_data = []
                graph_batch = []
                edge_att = []
                edge_ind = []
                target_qual = []
                if layer == 0:
                    if initial_state is not None:
                        layer_init_state = initial_state
                    else:
                        layer_init_state = model.boundary_value * torch.ones((data[0].shape[0],), dtype=torch.float32,
                                                                             device=device)

                # determine the initial heat state for the layer by transfering the final state of the previous layer, taking into account the
                # diffusion in the time passed during recoating
                else:
                    time_delta = (data[4][0] - prev_last_time) / 1000
                    state = model(prev_distances, prev_densities, prev_boundary, state, time_delta, transformed_tensor, y)
                    state = state.detach().view(-1)
                    layer_init_state = transfer_state(prev_vertices, state, data[0])

                prev_vertices = data[0]
                prev_distances = data[1]
                prev_densities = data[2]
                prev_boundary = data[3]
                prev_last_time = data[4][-1]

                state = layer_init_state

                # sample the evaluation time-steps (either random or n-step)

                # jumps = np.random.poisson(lambd,len(data[5])) + 1
                # eval_ixs = np.cumsum(jumps)
                # eval_ixs = eval_ixs[np.where(eval_ixs<len(data[5]))].tolist()
                # if eval_ixs[0] > 0:
                #     eval_ixs = [0,] + eval_ixs

                if whole_layer:
                    eval_ixs = [0, len(data[5]) - 1]
                else:
                    eval_ixs = np.arange(0, len(data[5]), lambd).tolist()
                    if eval_ixs[-1] < len(data[5]) - 1:
                        eval_ixs.append(len(data[5]) - 1)

                X = [(data[1], data[2], data[3], data[5][eval_ixs[i]],
                      (data[4][eval_ixs[i + 1]] - data[4][eval_ixs[i]]) / 1000, data[6][eval_ixs[i + 1]]) for i in
                     range(len(eval_ixs) - 1)]
                Y = [data[5][eval_ixs[i]] for i in range(1, len(eval_ixs))]
                count = 0
                for (distances, densities, boundary, surface_temp, time_delta, laser_dist), y in zip(X, Y):



                    # assign the surface temperature values
                    state[surface_temp.indices()[0]] = 0.
                    state = state + surface_temp.to_dense()

                    # Data Augmentation ##########################
                    transformed_tensor = random_transform(data[8])
                    #output_dim = y.values().size(dim=0)
                    #evaluate the model and loss function
                    pred, conn_loss, diss_loss, diss_vec, laser_heat, edge_attr, edge_index = model(distances.to(device), densities.to(device),
                                                                             boundary.to(device), state.to(device),
                                                                             time_delta.to(device), transformed_tensor.to(device), y,
                                                                             laser_dist.to(device), fit=True)
                    pred_loss = loss_fn(pred[y.indices()].view(-1, 1), y.values().view(-1, 1))
                    #greyscale_loss = loss_fn(pred_grey, transformed_tensor)

                    ################################# generate batch ###############################
                    graph_pred_temp.append(pred)
                    grayscale_data.append(transformed_tensor.squeeze(1))
                    graph_layer = torch.ones(pred.shape)*count
                    graph_batch.append(graph_layer)
                    edge_ind.append(edge_index)
                    edge_att.append(edge_attr)
                    if 0 <= layer <= 40:
                        lab = 0
                    elif 41 <= layer <= 110:
                        lab = 1
                    else:
                        lab = 2
                    target_qual.append(lab)
                    #high_qual = torch.tensor([1., 0., 0.])
                    #high_qual = high_qual.unsqueeze(0)


                    #print(f"quality_pred shape: {quality_pred.shape}")  # Expected: [1, 3]
                    #print(f"high_qual shape: {high_qual.shape}") # Expected: [1, 3]
                    #target = torch.tensor([lab]).to(quality_pred.device)
                    #qual_loss = bce_loss_fn(quality_pred, target)
                    #predictions = torch.argmax(quality_pred, dim=1)
                    #print(qual_loss.detach().numpy())
                    #total += target.size(0)
                    #correct += (predictions == target).sum().item()
                    #print(quality_pred.detach().numpy())

                    # calculate the regularizing loss functions
                    pred = pred - laser_heat
                    neighbor_temp = (torch.sign(distances).to_dense() * pred.view(-1).to_dense())
                    max_principle_violation = (torch.relu(pred.view(-1) - neighbor_temp.max(dim=1).values)).where(
                        torch.logical_and(pred > state, boundary.sum(dim=1) == 0),
                        torch.zeros_like(pred)).square().mean()
                    min_principle_violation = (torch.relu(neighbor_temp.min(dim=1).values - pred.view(-1))).where(
                        torch.logical_and(pred > state, boundary.sum(dim=1) == 0),
                        torch.zeros_like(pred)).square().mean()
                    energy_violation = torch.relu(
                        calculate_energy(pred, densities, boundary)  # energy of prediction
                        - calculate_energy(state, densities, boundary)  # energy of previous state
                    )
                    heat, heat_diss = calculate_heat(pred, densities, boundary, dissipation=diss_vec.squeeze())
                    total_heat_diff = ((heat - heat_diss) / state.sum() - 1).square()




                    #assemble the total loss
                    loss = pred_loss + \
                           alpha_conn * conn_loss + \
                           alpha_diss * diss_loss + \
                           alpha_energy * energy_violation + \
                           alpha_heat * total_heat_diff + \
                           alpha_max_min * (max_principle_violation + min_principle_violation)



                    # optimization step
                    #optim.zero_grad()
                    #loss.backward(retain_graph=True)
                    #optim.step()

                    # log the loss values
                    push_fwd_loss = loss_fn(state[y.indices()].view(-1, 1), y.values().view(-1, 1))
                    layer_loss += pred_loss.item() / (len(eval_ixs) - 1)
                    layer_const_loss += push_fwd_loss.item() / (len(eval_ixs) - 1)
                    layer_conn_loss += conn_loss.item() / (len(eval_ixs) - 1)
                    layer_diss_loss += diss_loss.item() / (len(eval_ixs) - 1)
                    layer_energy += (
                                                energy_violation + total_heat_diff + max_principle_violation + min_principle_violation).item() / (
                                                len(eval_ixs) - 1)

                    #layer_loss_greyscale += greyscale_loss.item() / (len(eval_ixs) - 1)
                    #layer_quality += qual_loss.item() / (len(eval_ixs) - 1)



                    count = count + 1
                    # use the predicted state for the next time-step
                    state = pred.detach().clip(0, t_max).view(-1)
                ###############################################################
                if testcount == 2:
                    k=0
                grayscale_data = torch.stack(grayscale_data, dim=0)

                temp_pred = torch.stack(graph_pred_temp, dim=0)
                grey_init = torch.ones(temp_pred.shape) * 0.5
                grey_init = grey_init.T




                graph_batch = torch.stack(graph_batch, dim=0)
                edge_att = torch.stack(edge_att, dim=0)
                edge_ind = torch.stack(edge_ind, dim=0)
                emb, pred_grey = autoencoder(grayscale_data)

                # Hauptkomponenten berechnen
                def principal_components(encoded_data, n_comp):
                    encoded_data = encoded_data.detach().numpy()
                    samples = encoded_data.shape
                    encoded_data_reshaped = encoded_data.reshape((samples[0], samples[1] * samples[2] * samples[3]))
                    transposed_encoded_data = encoded_data_reshaped.T
                    pca = PCA(n_components=n_comp)
                    pca.fit(transposed_encoded_data)
                    return pca.components_

                output_dim = y.values().size(dim=0)
                components = principal_components(emb, output_dim)
                components = torch.tensor(components)
                grey_init[y.indices(), :] = components
                #components = torch.flatten(components)[:, None]
                grey_init = torch.flatten(grey_init)



                temp_pred = torch.flatten(temp_pred)
                graph_batch = torch.flatten(graph_batch)
                graph_batch= graph_batch.to(torch.int64)

                #print(graph_batch)  # Should look like [0, 0, ..., 1, 1, ..., 9, 9] for 10 graphs.
                #print(graph_batch.shape)  # Should match the number of nodes.
                edge_att = torch.flatten(edge_att)[:, None]
                edge_ind = edge_ind.permute(0, 2, 1)
                edge_ind = edge_ind.reshape(-1, 2)


                #grey_init = torch.ones(temp_pred.size(dim=0)) * 0.5

                multifeat = torch.stack((temp_pred, grey_init))
                multifeat = multifeat.T
                #multifeat = multifeat.T
                #multifeat[y.indices(), 1] = components
                edge_ind = edge_ind.T

                pred_gT = graph_transformer(multifeat, edge_ind, edge_att, graph_batch)
                target = torch.tensor(target_qual).to(pred_gT.device)
                qual_loss = bce_loss_fn(pred_gT, target)

                # optimization step
                optim_MPGT.zero_grad()
                qual_loss.backward()
                optim_MPGT.step()

                layer_avg_steps += len(eval_ixs) - 1
                #accuracy = correct / total
                if pbar:
                    if layer_const_loss != 0.:
                        itr.set_postfix({'l_T': layer_loss,
                                         'rel': (layer_const_loss - layer_loss) / layer_const_loss,
                                         'l_reg': (layer_conn_loss + layer_diss_loss),
                                         'l_E': layer_energy,
                                         't_min': state.min().item(),
                                         't_max': state.max().item(),
                                        'l_grey': layer_loss_greyscale,
                                        'qual': layer_quality})


                # energy_loss.append(layer_energy)
                depth_avg_loss += layer_loss
                depth_fwd_loss += layer_const_loss

            if not depth_fwd_loss == 0:
                if pbar:
                    print(depth_avg_loss / depth, (depth_fwd_loss - depth_avg_loss) / depth_fwd_loss)
                else:
                    ditr.set_postfix(
                        {'l_T': depth_avg_loss / depth, 'rel': (depth_fwd_loss - depth_avg_loss) / depth_fwd_loss})
            if save_depth:
                scheduler.step()
                if depth_avg_loss < best_loss:
                    best_loss = depth_avg_loss
                    print('saving model state.')
                    model.save(save_path, compiled=True, override=True)

        # plot_state(data[0].detach().numpy(), state.detach().numpy(),data[7], set_limits=set_plot_limits, save_path=f'plots/depth_{depth}_plot{dix}.svg')
        print('parameters:')
        print(model.diss_model.coefs)
        print(model.laser_model.intensity.item(), ', ', model.laser_model.decay.item(), sep='')

    return history

def develop_layers_mulitmodal(model,
                   n_layers=110,
                   print_layers=[5, 10, 25, 50, 100, 150, 200, 250],
                   boundary_value=124.9,
                   use_data=None,
                   set_limits=True,
                   obj='pyramid_7',
                   vertex_multipliers=None,
                   plot_dir='./plots/layer_plots'):
    """
    evaluate the model and visualize the predicted heat states
    """

    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)
    optim = torch.optim.Adam(model.parameters(), lr=1e-5, betas=(.5, .999))
    loss_fn = nn.MSELoss()
    state = None
    history = []
    quality_label = []
    energy_loss = []
    itr = tqdm(range(n_layers))
    grey_loss_overall = []
    for layer in itr:
        if obj == 'pyramid_7':
            data = load_data_layer_multimodal(layer)
        else:
            data = load_data_layer_multimodal(layer, obj=obj, layers="0_to_10", min_layer=0, max_layer=10, vertex_multipliers = vertex_multipliers)

        if layer == 0:
            layer_init_state = boundary_value * torch.ones((data[0].shape[0],), dtype=torch.float32, device=device)
        else:
            time_delta = (data[4][0]-prev_last_time)/1000
            state, pred_qual, grey = model(prev_distances, prev_densities, prev_boundary, state, time_delta, transformed_tensor, y)
            state = state.detach().view(-1)
            layer_init_state = transfer_state(prev_vertices, state, data[0])
        prev_vertices = data[0]
        prev_distances = data[1]
        prev_densities = data[2]
        prev_boundary = data[3]
        prev_last_time = data[4][-1]
        layer_loss_greyscale = 0.

        state = layer_init_state

        eval_ixs = np.arange(0, len(data[5]), 1).tolist()
        if eval_ixs[-1] < len(data[5])-1:
            eval_ixs.append(len(data[5]) - 1)

        X = [(data[1], data[2], data[3], data[5][eval_ixs[i]], (data[4][eval_ixs[i+1]]-data[4][eval_ixs[i]])/1000, data[6][eval_ixs[i+1]]) for i in range(len(eval_ixs)-1)]
        Y = [data[5][eval_ixs[i]] for i in range(1, len(eval_ixs))]

        X_1 = []
        Y_1 = []
        X_1.append(X[0])
        X_1.append(X[-1])
        Y_1.append(Y[0])
        Y_1.append(Y[-1])

        layer_pred_quality = 0.
        layer_pred_grey_quality = 0.
        layer_surface_grey = 0.
        layer_energy = 0.
        counter = 0
        correct = 0
        total = 0
        for (distances, densities, boundary, surface_temp, time_delta, laser_dist), y in zip(X,Y):
            if use_data is None or layer in use_data:
                state[surface_temp.indices()[0]] = 0.
                state = state + surface_temp.to_dense()

            transformed_tensor = random_transform(data[8])
            pred, quality_pred, pred_grey = model(distances, densities, boundary, state, time_delta, transformed_tensor, y, laser_dist)
            #pred, conn_loss, diss_loss, diss_vec, laser_heat, pred_grey, quality_pred  = model(distances.to(device), densities.to(device),
                                                                     #boundary.to(device), state.to(device),
                                                                     #time_delta.to(device), transformed_tensor.to(device), y, laser_dist.to(device),
                                                                     #fit=True)

            if 0 <= layer <= 50:
                target = 0
            elif 51 <= layer <= 159:
                target = 1
            else:
                target = 2

            target = torch.tensor([target]).to(quality_pred.device)
            predictions = torch.argmax(quality_pred, dim=1)
            print(predictions)
            total += target.size(0)
            correct += (predictions == target).sum().item()
            accuracy = correct / total
            print(f"Test Accuracy: {accuracy * 100:.2f}%")

            #quality_label.append(high_qual.detach().numpy())
            #history.append(quality_pred.detach().numpy())
            ########################################################################################
            #pred_quality = torch.mean(torch.square(pred-state)[surface_temp.indices()[0]])
            #####################################
            #pred_grey_quality = torch.mean(torch.binary_cross_entropy_with_logits(pred_grey, data[8]))
            #layer_grey_p = torch.binary_cross_entropy_with_logits(pred_grey, data[8])
            #pred_quality = torch.corrcoef(pred_true)[1, 0]
            #pred_quality = torch.nan_to_num(pred_quality).item()
            #layer_pred_quality += pred_quality/(len(data[5])-1)
            #layer_pred_quality = quality_pred
            #layer_pred_grey_quality += pred_grey_quality/(len(data[5])-1)
            #layer_surface_grey += layer_grey_p/(len(data[5])-1)
            greyscale_loss = loss_fn(pred_grey, transformed_tensor)
            loss = greyscale_loss

            # optimization step
            loss.backward()
            optim.step()
            #layer_loss_greyscale += greyscale_loss.item() / (len(eval_ixs) - 1)

            itr.set_postfix({'l_grey': greyscale_loss})

            state = pred.detach().clip(0, 1e3).view(-1)
            counter = counter + 1

        #history.append(layer_pred_quality)

        ##################################################
        #grey_loss_overall.append(layer_pred_grey_quality)
        #energy_loss.append(layer_energy) # weg
        #itr.set_postfix({'corr': layer_pred_quality})
        counter = 0
        #if (layer+1) in print_layers:

            #plot_surface_greyscale(layer_surface_grey.detach().numpy(), save_path=f'{plot_dir}/surface_layer_greyscale{layer+1}.pdf')
            #plot_state(data[0].detach().numpy(), state.detach().numpy(), data[7], set_limits=set_limits, C_to_K=set_limits, show=False, save_path=f'{plot_dir}/layer_{layer+1}_same_range.pdf')
            #layer_temp = load_surface_temperatures(layer, obj=obj, start=(571 if obj=='pyramid_7' else 0))[-1]
            #plot_surface(layer_temp, save_path=f'{plot_dir}/surface_data_layer_{layer+1}.pdf', set_limits=set_limits, C_to_K=set_limits)
            #plot_surface_state(data[0].detach().numpy(), state.detach().numpy(), save_path=f'{plot_dir}/surface_layer_{layer+1}.pdf', set_limits=set_limits, C_to_K=set_limits,vertex_multipliers=vertex_multipliers)


    #return history, energy_loss, grey_loss_overall #weg
    return history, quality_label # weg

def train_MPGT(model,
                   n_layers=200,
                   print_layers=[5, 10, 25, 50, 100, 150, 200, 250],
                   boundary_value=124.9,
                   use_data=None,
                   set_limits=True,
                   obj='pyramid_7',
                   vertex_multipliers=None,
                   plot_dir='./plots/layer_plots'):



    graph_transformer = MultiModalGraphTransformer()
    autoencoder = AutoencoderGreyscale_pretrained()
    autoencoder.load_state_dict(torch.load("best_autoencoder.pth"))
    graph_transformer.load_state_dict(torch.load("best_graph_transformer.pth"))
    optim_MPGT = torch.optim.Adam(graph_transformer.parameters(), lr=1e-3)
    bce_loss_fn = nn.CrossEntropyLoss()

    testcount = 0
    epochs = 50
    itr = tqdm(range(n_layers))
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        train_loss = 0.0
        #MultiFeat = []
        #GraphBatch = []
        #EdgeIndex = []
        #EdgeAtrribute = []
        MultiFeat = torch.load('MultiFeat_pyr9.pt')
        EdgeIndex = torch.load('EdgeIndex_pyr9.pt')
        EdgeAtrribute = torch.load('EdgeAttribute_pyr9.pt')
        GraphBatch = torch.load('GraphBatch_pyr9.pt')
        for layer in itr:
            testcount = testcount + 1
            graph_pred_temp = []
            grayscale_data = []
            graph_batch = []
            edge_att = []
            edge_ind = []
            target_qual = []

            if obj == 'pyramid_7':
                data = load_data_layer_multimodal(layer)
            else:
                data = load_data_layer_multimodal(layer, obj=obj, layers="0_to_10", min_layer=0, max_layer=10,
                                                  vertex_multipliers=vertex_multipliers)

            if layer == 0:
                layer_init_state = boundary_value * torch.ones((data[0].shape[0],), dtype=torch.float32, device=device)
            else:
                time_delta = (data[4][0] - prev_last_time) / 1000
                state = model(prev_distances, prev_densities, prev_boundary, state, time_delta)
                state = state.detach().view(-1)
                layer_init_state = transfer_state(prev_vertices, state, data[0])
            prev_vertices = data[0]
            prev_distances = data[1]
            prev_densities = data[2]
            prev_boundary = data[3]
            prev_last_time = data[4][-1]
            layer_loss_greyscale = 0.

            state = layer_init_state

            eval_ixs = np.arange(0, len(data[5]), 1).tolist()
            if eval_ixs[-1] < len(data[5]) - 1:
                eval_ixs.append(len(data[5]) - 1)

            X = [(data[1], data[2], data[3], data[5][eval_ixs[i]], (data[4][eval_ixs[i + 1]] - data[4][eval_ixs[i]]) / 1000,
                  data[6][eval_ixs[i + 1]]) for i in range(len(eval_ixs) - 1)]
            Y = [data[5][eval_ixs[i]] for i in range(1, len(eval_ixs))]

            X_1 = []
            Y_1 = []
            X_1.append(X[0])
            X_1.append(X[-1])
            Y_1.append(Y[0])
            Y_1.append(Y[-1])

            layer_pred_quality = 0.
            layer_pred_grey_quality = 0.
            layer_surface_grey = 0.
            layer_energy = 0.
            counter = 0
            correct = 0
            total = 0
            count = 0
            for (distances, densities, boundary, surface_temp, time_delta, laser_dist), y in zip(X, Y):
                # assign the surface temperature values
                state[surface_temp.indices()[0]] = 0.
                state = state + surface_temp.to_dense()

                # Data Augmentation ##########################
                transformed_tensor = random_transform(data[8])
                #output_dim = y.values().size(dim=0)
                #evaluate the model and loss function
                pred, conn_loss, diss_loss, diss_vec, laser_heat, edge_attr, edge_index = model(distances.to(device), densities.to(device),
                                                                         boundary.to(device), state.to(device),
                                                                         time_delta.to(device),
                                                                         laser_dist.to(device), fit=True)

                #greyscale_loss = loss_fn(pred_grey, transformed_tensor)

                ################################# generate batch ###############################
                graph_pred_temp.append(pred)
                grayscale_data.append(transformed_tensor.squeeze(1))
                graph_layer = torch.ones(pred.shape)*count
                graph_batch.append(graph_layer)
                edge_ind.append(edge_index)
                edge_att.append(edge_attr)
                if 0 <= layer <= 40:
                    lab = 0
                elif 41 <= layer <= 110:
                    lab = 1
                else:
                    lab = 2
                target_qual.append(lab)
                #high_qual = torch.tensor([1., 0., 0.])
                #high_qual = high_qual.unsqueeze(0)


                #print(f"quality_pred shape: {quality_pred.shape}")  # Expected: [1, 3]
                #print(f"high_qual shape: {high_qual.shape}") # Expected: [1, 3]
                #target = torch.tensor([lab]).to(quality_pred.device)
                #qual_loss = bce_loss_fn(quality_pred, target)
                #predictions = torch.argmax(quality_pred, dim=1)
                #print(qual_loss.detach().numpy())
                #total += target.size(0)
                #correct += (predictions == target).sum().item()
                #print(quality_pred.detach().numpy())

                # calculate the regularizing loss functions
                pred = pred - laser_heat
                neighbor_temp = (torch.sign(distances).to_dense() * pred.view(-1).to_dense())
                max_principle_violation = (torch.relu(pred.view(-1) - neighbor_temp.max(dim=1).values)).where(
                    torch.logical_and(pred > state, boundary.sum(dim=1) == 0),
                    torch.zeros_like(pred)).square().mean()
                min_principle_violation = (torch.relu(neighbor_temp.min(dim=1).values - pred.view(-1))).where(
                    torch.logical_and(pred > state, boundary.sum(dim=1) == 0),
                    torch.zeros_like(pred)).square().mean()
                energy_violation = torch.relu(
                    calculate_energy(pred, densities, boundary)  # energy of prediction
                    - calculate_energy(state, densities, boundary)  # energy of previous state
                )
                heat, heat_diss = calculate_heat(pred, densities, boundary, dissipation=diss_vec.squeeze())
                total_heat_diff = ((heat - heat_diss) / state.sum() - 1).square()

                # use the predicted state for the next time-step
                state = pred.detach().clip(0, 1e3).view(-1)
                count = count + 1
            ###############################################################
            if testcount == 90:
                k=0
            grayscale_data = torch.stack(grayscale_data, dim=0)
            ############################# PCA tuning##################
            n_comp = y.values().size(dim=0)
            last_index = len(graph_batch) - 1
            """if last_index <= n_comp:
                # Define the target size (e.g., extend to 10 tensors in the batch dimension)
                target_size = n_comp + 1

                # Randomly repeat existing data
                num_repeats = target_size // grayscale_data.size(0) + 1  # Calculate required repeats
                extended_data = grayscale_data.repeat((num_repeats, 1, 1, 1))  # Repeat data along batch dimension

                # Trim to the target size
                extended_data = extended_data[:target_size]
            else: extended_data = grayscale_data"""
            ###################################################################################################
            if layer >= 126:
                temp_pred = torch.stack(graph_pred_temp, dim=0)
                temp_pred = temp_pred/700
                grey_init = torch.ones(temp_pred.shape) * 0.5
                grey_init = grey_init.T

                graph_batch = torch.stack(graph_batch, dim=0)
                edge_att = torch.stack(edge_att, dim=0)
                edge_ind = torch.stack(edge_ind, dim=0)
                #emb, pred_grey = autoencoder(grayscale_data)
                emb, pred_grey = autoencoder(grayscale_data)



                #Hauptkomponenten berechnen
                def principal_components(encoded_data, n_comp):

                    encoded_data = encoded_data.detach().numpy()
                    samples = encoded_data.shape
                    encoded_data_reshaped = encoded_data.reshape((samples[0], samples[1] * samples[2] * samples[3]))
                    transposed_encoded_data = encoded_data_reshaped.T
                    pca = PCA(n_components=n_comp)
                    pca.fit(transposed_encoded_data)
                    return pca.components_

                output_dim = y.values().size(dim=0)
                #components = principal_components(emb, output_dim)
                rp = GaussianRandomProjection(n_components=output_dim, random_state=42)
                encoded_data_reshaped = emb.view(emb.size(0), -1).detach().cpu().numpy()
                components = rp.fit_transform(encoded_data_reshaped)
                components = components.T
                components = torch.tensor(components)
                graph_batch_int = graph_batch.to(torch.int64)
                timepoints = torch.max(graph_batch_int).item() ########## hier weitermachen
                grey_init[y.indices(), :] = components[:, 0:timepoints+1].float()
                #components = torch.flatten(components)[:, None]
                grey_init = torch.flatten(grey_init)



                temp_pred = torch.flatten(temp_pred)
                graph_batch = torch.flatten(graph_batch)
                graph_batch= graph_batch.to(torch.int64)

                #print(graph_batch)  # Should look like [0, 0, ..., 1, 1, ..., 9, 9] for 10 graphs.
                #print(graph_batch.shape)  # Should match the number of nodes.
                edge_att = torch.flatten(edge_att)[:, None]
                edge_ind = edge_ind.permute(0, 2, 1)
                edge_ind = edge_ind.reshape(-1, 2)


                #grey_init = torch.ones(temp_pred.size(dim=0)) * 0.5

                multifeat = torch.stack((temp_pred, grey_init))
                multifeat = multifeat.T
                #multifeat = multifeat.T
                #multifeat[y.indices(), 1] = components
                edge_ind = edge_ind.T

                ################################## data save ############################


                #########################################################################
                MultiFeat.append(multifeat)
                EdgeIndex.append(edge_ind)
                EdgeAtrribute.append(edge_att)
                GraphBatch.append(graph_batch)
                pred_gT = graph_transformer(multifeat, edge_ind, edge_att, graph_batch)
                target = torch.tensor(target_qual).to(pred_gT.device)
                qual_loss = bce_loss_fn(pred_gT, target)

                # optimization step
                optim_MPGT.zero_grad()
                qual_loss.backward()
                optim_MPGT.step()

                train_loss += qual_loss.item()
        torch.save(MultiFeat, 'MultiFeat_pyr9.pt')
        torch.save(EdgeIndex, 'EdgeIndex_pyr9.pt')
        torch.save(EdgeAtrribute, 'EdgeAttribute_pyr9.pt')
        torch.save(GraphBatch, 'GraphBatch_pyr9.pt')
        train_losses.append(train_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}")
        #torch.save(graph_transformer.state_dict(), "best_graph_transformer.pth")  # Save best model
        #accuracy = correct / total

    return train_losses


def train_MPGT_eff(n_layers=180):


    graph_transformer = MultiModalGraphTransformer()
    # graph_transformer.load_state_dict(torch.load("best_graph_transformer.pth"))
    optim_MPGT = torch.optim.Adam(graph_transformer.parameters(), lr=1e-3)
    bce_loss_fn = nn.CrossEntropyLoss()
    mulitfeat = torch.load('MultiFeat_pyr9.pt')

    edge_ind = torch.load('EdgeIndex_pyr9.pt')
    edge_att = torch.load('EdgeAttribute_pyr9.pt')
    graph_batch = torch.load('GraphBatch_pyr9.pt')
    target = torch.load('TARGET_pyr9_140125')
    # Calculate the total number of values
    total_count = sum(len(entry) for entry in target)

    print("Total count:", total_count)

    itr = tqdm(range(n_layers))
    testcount = 0
    epochs = 100
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        train_loss = 0.0

        for layer in itr:
            pred_gT = graph_transformer(mulitfeat[layer], edge_ind[layer], edge_att[layer], graph_batch[layer])
            qual_loss = bce_loss_fn(pred_gT, target[layer])

            # optimization step
            optim_MPGT.zero_grad()
            qual_loss.backward()
            optim_MPGT.step()

            train_loss += qual_loss.item()

        train_losses.append(train_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}")
        print('TRAIN')
        #torch.save(graph_transformer.state_dict(), "best_graph_transformer.pth")  # Save best model
        # accuracy = correct / total

    return train_losses

def quality_prediction(model,
                   n_layers=100,
                   print_layers=[5, 10, 25, 50, 100, 150, 200, 250],
                   boundary_value=124.9,
                   use_data=None,
                   set_limits=True,
                   obj='pyramid_7',
                   vertex_multipliers=None,
                   plot_dir='./plots/layer_plots'):



    graph_transformer = MultiModalGraphTransformer()
    autoencoder = AutoencoderGreyscale_pretrained()
    autoencoder.load_state_dict(torch.load("best_autoencoder.pth"))
    graph_transformer.load_state_dict(torch.load("best_graph_transformer_update.pth"))
    training_data_counter = 13063.

    testcount = 0
    epochs = 50
    itr = tqdm(range(n_layers))
    train_losses = []
    val_losses = []
    correct = 0
    total = 0
    all_predictions = torch.load('all_predictions')
    all_targets = torch.load('all_targets')
    #all_predictions = []
    #all_targets = []
    for layer in itr:
        testcount = testcount + 1
        graph_pred_temp = []
        grayscale_data = []
        graph_batch = []
        edge_att = []
        edge_ind = []
        target_qual = []

        if obj == 'pyramid_7':
            data = load_data_layer_multimodal(layer)
        else:
            data = load_data_layer_multimodal(layer, obj=obj, layers="0_to_10", min_layer=0, max_layer=10,
                                              vertex_multipliers=vertex_multipliers)

        if layer == 0:
            layer_init_state = boundary_value * torch.ones((data[0].shape[0],), dtype=torch.float32, device=device)
        else:
            time_delta = (data[4][0] - prev_last_time) / 1000
            state = model(prev_distances, prev_densities, prev_boundary, state, time_delta)
            state = state.detach().view(-1)
            layer_init_state = transfer_state(prev_vertices, state, data[0])
        prev_vertices = data[0]
        prev_distances = data[1]
        prev_densities = data[2]
        prev_boundary = data[3]
        prev_last_time = data[4][-1]
        layer_loss_greyscale = 0.

        state = layer_init_state

        eval_ixs = np.arange(0, len(data[5]), 1).tolist()
        if eval_ixs[-1] < len(data[5]) - 1:
            eval_ixs.append(len(data[5]) - 1)

        X = [(data[1], data[2], data[3], data[5][eval_ixs[i]], (data[4][eval_ixs[i + 1]] - data[4][eval_ixs[i]]) / 1000,
              data[6][eval_ixs[i + 1]]) for i in range(len(eval_ixs) - 1)]
        Y = [data[5][eval_ixs[i]] for i in range(1, len(eval_ixs))]

        X_1 = []
        Y_1 = []
        X_1.append(X[0])
        X_1.append(X[-1])
        Y_1.append(Y[0])
        Y_1.append(Y[-1])

        layer_pred_quality = 0.
        layer_pred_grey_quality = 0.
        layer_surface_grey = 0.
        layer_energy = 0.
        counter = 0
        count = 0
        for (distances, densities, boundary, surface_temp, time_delta, laser_dist), y in zip(X, Y):
            training_data_counter += 1
            # assign the surface temperature values
            state[surface_temp.indices()[0]] = 0.
            state = state + surface_temp.to_dense()

            # Data Augmentation ##########################
            transformed_tensor = random_transform(data[8])
            #output_dim = y.values().size(dim=0)
            #evaluate the model and loss function
            pred, conn_loss, diss_loss, diss_vec, laser_heat, edge_attr, edge_index = model(distances.to(device), densities.to(device),
                                                                     boundary.to(device), state.to(device),
                                                                     time_delta.to(device),
                                                                     laser_dist.to(device), fit=True)

            #greyscale_loss = loss_fn(pred_grey, transformed_tensor)

            ################################# generate batch ###############################
            graph_pred_temp.append(pred)
            grayscale_data.append(transformed_tensor.squeeze(1))
            graph_layer = torch.ones(pred.shape)*count
            graph_batch.append(graph_layer)
            edge_ind.append(edge_index)
            edge_att.append(edge_attr)
            """if 0 <= layer <= 40:
                lab = 0
            elif 41 <= layer <= 110:
                lab = 1
            else:
                lab = 2"""
            if 0 <= layer <= 100:
                lab = 0
            elif 61 <= layer <= 130:
                lab = 1
            else:
                lab = 2
            target_qual.append(lab)

            #high_qual = torch.tensor([1., 0., 0.])
            #high_qual = high_qual.unsqueeze(0)


            #print(f"quality_pred shape: {quality_pred.shape}")  # Expected: [1, 3]
            #print(f"high_qual shape: {high_qual.shape}") # Expected: [1, 3]
            #target = torch.tensor([lab]).to(quality_pred.device)
            #qual_loss = bce_loss_fn(quality_pred, target)
            #predictions = torch.argmax(quality_pred, dim=1)
            #print(qual_loss.detach().numpy())
            #total += target.size(0)
            #correct += (predictions == target).sum().item()
            #print(quality_pred.detach().numpy())

            # calculate the regularizing loss functions
            pred = pred - laser_heat
            neighbor_temp = (torch.sign(distances).to_dense() * pred.view(-1).to_dense())
            max_principle_violation = (torch.relu(pred.view(-1) - neighbor_temp.max(dim=1).values)).where(
                torch.logical_and(pred > state, boundary.sum(dim=1) == 0),
                torch.zeros_like(pred)).square().mean()
            min_principle_violation = (torch.relu(neighbor_temp.min(dim=1).values - pred.view(-1))).where(
                torch.logical_and(pred > state, boundary.sum(dim=1) == 0),
                torch.zeros_like(pred)).square().mean()
            energy_violation = torch.relu(
                calculate_energy(pred, densities, boundary)  # energy of prediction
                - calculate_energy(state, densities, boundary)  # energy of previous state
            )
            heat, heat_diss = calculate_heat(pred, densities, boundary, dissipation=diss_vec.squeeze())
            total_heat_diff = ((heat - heat_diss) / state.sum() - 1).square()

            # use the predicted state for the next time-step
            state = pred.detach().clip(0, 1e3).view(-1)
            count = count + 1
        ###############################################################
        if testcount == 90:
            k=0
        grayscale_data = torch.stack(grayscale_data, dim=0)
        ############################# PCA tuning##################
        n_comp = y.values().size(dim=0)
        last_index = len(graph_batch) - 1
        if last_index <= n_comp:
            # Define the target size (e.g., extend to 10 tensors in the batch dimension)
            target_size = n_comp + 1

            # Randomly repeat existing data
            num_repeats = target_size // grayscale_data.size(0) + 1  # Calculate required repeats
            extended_data = grayscale_data.repeat((num_repeats, 1, 1, 1))  # Repeat data along batch dimension

            # Trim to the target size
            extended_data = extended_data[:target_size]
        else: extended_data = grayscale_data
        ###################################################################################################

        temp_pred = torch.stack(graph_pred_temp, dim=0)
        temp_pred = temp_pred/700
        grey_init = torch.ones(temp_pred.shape) * 0.5
        grey_init = grey_init.T

        graph_batch = torch.stack(graph_batch, dim=0)
        edge_att = torch.stack(edge_att, dim=0)
        edge_ind = torch.stack(edge_ind, dim=0)
        #emb, pred_grey = autoencoder(grayscale_data)
        emb, pred_grey = autoencoder(extended_data)



        #Hauptkomponenten berechnen
        def principal_components(encoded_data, n_comp):

            encoded_data = encoded_data.detach().numpy()
            samples = encoded_data.shape
            encoded_data_reshaped = encoded_data.reshape((samples[0], samples[1] * samples[2] * samples[3]))
            transposed_encoded_data = encoded_data_reshaped.T
            pca = PCA(n_components=n_comp)
            pca.fit(transposed_encoded_data)
            return pca.components_

        output_dim = y.values().size(dim=0)

        if layer >= 126:
            emb, pred_grey = autoencoder(grayscale_data)
            rp = GaussianRandomProjection(n_components=output_dim, random_state=42)
            encoded_data_reshaped = emb.view(emb.size(0), -1).detach().cpu().numpy()
            components = rp.fit_transform(encoded_data_reshaped)
            components = components.T
        else:
            components = principal_components(emb, output_dim)
        components = torch.tensor(components)
        graph_batch_int = graph_batch.to(torch.int64)
        timepoints = torch.max(graph_batch_int).item() ########## hier weitermachen
        grey_init[y.indices(), :] = components[:, 0:timepoints+1].float()
        #components = torch.flatten(components)[:, None]
        grey_init = torch.flatten(grey_init)



        temp_pred = torch.flatten(temp_pred)
        graph_batch = torch.flatten(graph_batch)
        graph_batch= graph_batch.to(torch.int64)

        #print(graph_batch)  # Should look like [0, 0, ..., 1, 1, ..., 9, 9] for 10 graphs.
        #print(graph_batch.shape)  # Should match the number of nodes.
        edge_att = torch.flatten(edge_att)[:, None]
        edge_ind = edge_ind.permute(0, 2, 1)
        edge_ind = edge_ind.reshape(-1, 2)


        #grey_init = torch.ones(temp_pred.size(dim=0)) * 0.5

        multifeat = torch.stack((temp_pred, grey_init))
        multifeat = multifeat.T
        #multifeat = multifeat.T
        #multifeat[y.indices(), 1] = components
        edge_ind = edge_ind.T

        pred_gT = graph_transformer(multifeat, edge_ind, edge_att, graph_batch)
        target = torch.tensor(target_qual).to(pred_gT.device)

        #target = torch.tensor([lab]).to(quality_pred.device)
        predictions = torch.argmax(pred_gT, dim=1)
        #total += target.size(0)
        #correct += (predictions == target).sum().item()
        #accuracy = correct / total
        #print(f"Test Accuracy: {accuracy * 100:.2f}%")
        # Predictions and targets (convert to numpy)
        all_predictions.extend(predictions.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

        # Class-wise metrics
        class_precision = precision_score(all_targets, all_predictions, average=None, zero_division=0)
        class_recall = recall_score(all_targets, all_predictions, average=None, zero_division=0)
        class_f1 = f1_score(all_targets, all_predictions, average=None, zero_division=0)

        # weighted average metrics
        precision = precision_score(all_targets, all_predictions, average='weighted')
        recall = recall_score(all_targets, all_predictions, average='weighted')
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        accuracy = accuracy_score(all_targets, all_predictions)

        print(f"Class-wise Precision: {class_precision}")
        print(f"Class-wise Recall: {class_recall}")
        print(f"Class-wise F1 Score: {class_f1}")
        print(f"Accuracy: {accuracy * 100:.2f}%")

        # Print metrics
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall: {recall * 100:.2f}%")
        print(f"F1 Score: {f1 * 100:.2f}%")
        print(training_data_counter)
        #qual_loss = bce_loss_fn(pred_gT, target)



    #print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}")
    #torch.save(graph_transformer.state_dict(), "best_graph_transformer.pth")  # Save best model
    #accuracy = correct / total
    torch.save(all_targets, 'all_targets')
    torch.save(all_predictions, 'all_predictions')

    return train_losses

def develop_layers_state(model, n_layers=500, boundary_value=124.9, use_data=None, obj=None, layers=None, return_all=False):

    if obj is None:
        obj = "pyramid_7"
    if layers is None:
        if obj == "pyramid_7":
            layers = "571_to_1079"
        elif obj == "pyramid_3":
            layers = "571_to_1064"
    
    state = None
    itr = tqdm(range(n_layers))
    
    data_losses = []
    consistency_losses = []
    heateq_losses = []
    layer_correlations = []
    
    loss_fn = nn.MSELoss()
    
    ts = int(dt.datetime.timestamp(dt.datetime.now()))
    dirname = f'eval_plots_{ts}'
    # os.mkdir(dirname)

    if return_all:
        states = []

    for layer in itr:
        if obj is None:
            data = load_data_layer(layer)
        else:
            data = load_data_layer(layer,obj=obj, layers=layers)
        
        layer_data_loss = 0.
        layer_consistency_loss = 0.
        layer_heateq_loss = 0.
    
        
        if layer==0:
            layer_init_state = boundary_value * torch.ones((data[0].shape[0],),dtype=torch.float32,device=device)
        else:
            time_delta=(data[4][0]-prev_last_time)/1000
            state = model(prev_distances, prev_densities, prev_boundary, state, time_delta).detach().view(-1)
            layer_init_state = transfer_state(prev_vertices,state,data[0])
        prev_vertices = data[0]
        prev_distances = data[1]
        prev_densities = data[2]
        prev_boundary = data[3]
        prev_last_time = data[4][-1]
        
        state = layer_init_state
        
        eval_ixs = np.arange(0,len(data[5]),1).tolist()
        if eval_ixs[-1] < len(data[5])-1:
            eval_ixs.append(len(data[5])-1)

        X = [(data[1], data[2], data[3], data[5][eval_ixs[i]], (data[4][eval_ixs[i+1]]-data[4][eval_ixs[i]])/1000, data[6][eval_ixs[i+1]]) for i in range(len(eval_ixs)-1)]
        Y = [data[5][eval_ixs[i]] for i in range(1,len(eval_ixs))]

        layer_pred_quality = 0.
        for (distances, densities, boundary, surface_temp, time_delta, laser_dist), y in zip(X,Y):
            if use_data is None or use_data[layer]:
                state[surface_temp.indices()[0]] = 0.
                state = state + surface_temp.to_dense()
            
            # pred = model(distances, densities, boundary, state, time_delta, laser_dist)
            pred = model(distances, densities, boundary, state, time_delta, laser_dist)
            state = pred.detach().clip(0,1e3).view(-1)
        if return_all:
            states.append(state.clone())
    if return_all:
        return states
    return state

def predict_layer(model,state,layer, prev_distances, prev_vertices, prev_densities, prev_boundary, prev_last_time, loss_fn = nn.MSELoss(), boundary_value=124.9, use_data=None, obj=None, layers=None, plot_states=False):
    if obj is None:
        data = load_data_layer(layer)
    else:
        data = load_data_layer(layer,obj=obj, layers=layers)
    
    layer_data_loss = 0.
    layer_consistency_loss = 0.
    layer_heateq_loss = 0.
    ######################
    energy_violation_loss = 0.
    conn_violation_loss = 0.
    diss_violation_loss = 0.
    ######################

    
    if layer==0:
        layer_init_state = boundary_value * torch.ones((data[0].shape[0],),dtype=torch.float32,device=device)
    else:
        time_delta=(data[4][0]-prev_last_time)/1000
        state = model(prev_distances, prev_densities, prev_boundary, state, time_delta).detach().view(-1)
        layer_init_state = transfer_state(prev_vertices,state,data[0])
    prev_vertices = data[0]
    prev_distances = data[1]
    prev_densities = data[2]
    prev_boundary = data[3]
    prev_last_time = data[4][-1]
    
    state = layer_init_state
    
    eval_ixs = np.arange(0,len(data[5]),1).tolist()
    if eval_ixs[-1] < len(data[5])-1:
        eval_ixs.append(len(data[5])-1)

    X = [(data[1], data[2], data[3], data[5][eval_ixs[i]], (data[4][eval_ixs[i+1]]-data[4][eval_ixs[i]])/1000, data[6][eval_ixs[i+1]]) for i in range(len(eval_ixs)-1)]
    Y = [data[5][eval_ixs[i]] for i in range(1,len(eval_ixs))]

    layer_pred_quality = 0.
    for ix, ((distances, densities, boundary, surface_temp, time_delta, laser_dist), y) in enumerate(zip(X,Y)):
        if use_data is None or layer in use_data:
            state[surface_temp.indices()[0]] = 0.
            state = state + surface_temp.to_dense()
        
        # pred = model(distances, densities, boundary, state, time_delta, laser_dist)
        pred, conn_loss, diss_loss, diss_vec, laser_heat = model(distances, densities, boundary, state, time_delta, laser_dist, fit=True)

        layer_data_loss += loss_fn(pred[y.indices()].view(-1,1),y.values().view(-1,1)).item()/(len(data[5])-1)

        pred = pred - laser_heat
        neighbor_temp = (torch.sign(distances).to_dense() * pred.view(-1).to_dense())
        max_principle_violation = (torch.relu(pred.view(-1)-neighbor_temp.max(dim=1).values)).where(torch.logical_and(pred > state,boundary.sum(dim=1)==0),torch.zeros_like(pred)).square().mean()
        min_principle_violation = (torch.relu(neighbor_temp.min(dim=1).values-pred.view(-1))).where(torch.logical_and(pred > state,boundary.sum(dim=1)==0),torch.zeros_like(pred)).square().mean()
        energy_violation = torch.relu(
                    calculate_energy(pred,densities,boundary) #energy of prediction
                    - calculate_energy(state,densities,boundary) #energy of previous state
                    )
        heat, heat_diss = calculate_heat(pred,densities,boundary, dissipation=diss_vec.squeeze())
        total_heat_diff = ((heat-heat_diss)/state.sum()-1).square()

        layer_consistency_loss += (conn_loss + diss_loss + total_heat_diff).item()/(len(data[5])-1)
        layer_heateq_loss += (energy_violation + max_principle_violation + min_principle_violation).item()/(len(data[5])-1)
        ####################################################################
        energy_violation_loss += (energy_violation).item()/(len(data[5])-1)
        conn_violation_loss += (conn_loss).item()/(len(data[5])-1)
        diss_violation_loss +=(diss_loss).item()/(len(data[5])-1)
        ####################################################################

        pred_true = torch.stack([((state-pred)[surface_temp.indices()[0]]),
                                    ((state-y)[surface_temp.indices()[0]])])
        pred_quality = torch.corrcoef(pred_true)[1,0]
        pred_quality = torch.nan_to_num(pred_quality).item()
        layer_pred_quality += pred_quality/(len(data[5])-1)

        state = pred.detach().clip(0,1e3).view(-1)
        if plot_states:
            plot_state(prev_vertices,state,np.array(data[7]),save_path=f'plots/animation_plots/{str(layer).zfill(4)}_{str(ix).zfill(3)}.png', show=False,plot_format='png',backend='agg')
    return state, (prev_vertices, prev_distances, prev_densities, prev_boundary, prev_last_time), (layer_data_loss,layer_consistency_loss, layer_heateq_loss, layer_pred_quality, conn_violation_loss, diss_violation_loss, energy_violation_loss)

        
    