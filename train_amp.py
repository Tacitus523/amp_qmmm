#!/usr/bin/env python

from Util import (
    load_parameters_file,
    dipole_loss,
    quadrupole_loss,
    SingleSystemOrcaXtbDataset,
    MultiSystemOrcaXtbDataset,
    INPUT_LAYOUT,
    instantiate_model,
    evaluate_on_dataset,
    set_model_dtype,
    assert_correct_dtype,
    batch_to_input,
    def_collate_fn,
)
import argparse
import sys
import os
import copy
import shutil
import torch
from torch.utils.data import DataLoader
import time
import numpy as np
from torchmetrics import MeanAbsoluteError as MAE
from torch.utils.tensorboard import SummaryWriter
import yaml
import random


def train_one_epoch(epoch, model, optimizer, loss_fn, training_loader, PARAMETERS):
    training_start = time.time()

    # MAEs
    training_mae_energy = MAE().to(model.device)
    training_mae_gradient_qm = MAE().to(model.device)
    training_mae_gradient_mm = MAE().to(model.device)
    if PARAMETERS['multi_loss']:
        training_mae_dipoles = MAE().to(model.device)
        training_mae_quadrupoles = MAE().to(model.device)

    # put model in train mode
    model.train()

    batch_idx = 0
    training_size = sum(len(dict_list) if not isinstance(dict_list, dict) else 1 for dict_list in training_loader)
    for dict_list_idx, dict_list in enumerate(training_loader):
        if isinstance(dict_list, dict):
            dict_list = [dict_list]
        for dict_idx, batch in enumerate(dict_list):
            if batch_idx % 100 == 0:
                print(f"Training: Epoch: {epoch}. Batch {batch_idx} / {training_size}")
            batch_idx += 1

            # transfer batch to GPU and prepare input
            for key, tensor in batch.items():
                batch[key] = tensor.to(model.device) if isinstance(tensor, torch.Tensor) else None
                if key == "qm_coordinates" or key == "mm_coordinates":
                    batch[key].requires_grad = True

            # check dtype
            for key in batch:
                if batch[key] is not None and batch[key].dtype != torch.int64:
                    if PARAMETERS["dtype"] == "float32":
                        assert batch[key].dtype == torch.float32
                    elif PARAMETERS["dtype"] == "float64":
                        assert batch[key].dtype == torch.float64
                    else:
                        print(f"Unsupported dtype: {PARAMETERS['dtype']}")
                        sys.exit(1)

            
            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # make prediction
            input = batch_to_input(batch)
            prediction, graph = model.forward_with_graph(input)
            
            # gradient prediction
            qm_gradients_pred = torch.autograd.grad(prediction, input[INPUT_LAYOUT["qm_coordinates"]], grad_outputs=torch.ones_like(prediction), retain_graph=True, create_graph=True)[0]
            mm_gradients_pred = torch.autograd.grad(prediction, input[INPUT_LAYOUT["mm_coordinates"]], grad_outputs=torch.ones_like(prediction), retain_graph=True, create_graph=True)[0]

            if PARAMETERS['delta_qm']:
                prediction += batch['delta_qm_energies']
                qm_gradients_pred += batch['delta_qm_gradients']
            elif PARAMETERS['delta_qmmm']:
                prediction += batch['delta_qm_energies']
                qm_gradients_pred += batch['delta_qm_gradients']
                mm_gradients_pred += batch['delta_mm_gradients']
            if PARAMETERS['multi_loss']:
                pred_dipole = model._molecular_dipole(graph)
                pred_quadrupole = model._molecular_quadrupole(graph)
                loss_dipoles = PARAMETERS['gamma'] * dipole_loss(batch['qm_dipoles'], pred_dipole, loss_fn)
                loss_quadrupoles = PARAMETERS['gamma'] * quadrupole_loss(batch['qm_quadrupoles'], pred_quadrupole, loss_fn)

            # compute loss (training)
            if not PARAMETERS["single_system"]:
                prediction = prediction - prediction[0]
            loss_potential = (1 - PARAMETERS['alpha']) * loss_fn(prediction, batch['qm_energies'])
            loss_qm_gradient = PARAMETERS['alpha'] * loss_fn(qm_gradients_pred, batch['qm_gradients'])
            loss_mm_gradient = PARAMETERS['alpha'] * PARAMETERS['beta'] * loss_fn(mm_gradients_pred, batch['mm_gradients'])        
            loss = loss_potential + loss_mm_gradient + loss_qm_gradient
            if PARAMETERS['multi_loss']:
                loss = loss + loss_dipoles + loss_quadrupoles

            # backprop and gradient descent
            loss.backward()

            for param in model.parameters():
                if torch.isnan(param.grad).any():
                    print("nan gradient found")
                    sys.exit(1)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=PARAMETERS['max_grad_norm'])
            
            optimizer.step()
            
            # compute MAE (training)
            training_mae_energy.update(batch['qm_energies'], prediction)
            training_mae_gradient_qm.update(batch['qm_gradients'], qm_gradients_pred)
            training_mae_gradient_mm.update(batch['mm_gradients'], mm_gradients_pred)
            if PARAMETERS['multi_loss']:
                training_mae_dipoles.update(batch['qm_dipoles'], pred_dipole)
                training_mae_quadrupoles.update(batch['qm_quadrupoles'], pred_quadrupole[:, [0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]])


    last_lr = optimizer.param_groups[0]['lr']
    
    # MAE
    maes = dict()
    maes["mae_energies"] = training_mae_energy.compute()
    maes["mae_qm_gradients"] = training_mae_gradient_qm.compute()
    maes["mae_mm_gradients"] = training_mae_gradient_mm.compute()
    if PARAMETERS['multi_loss']:
        maes["mae_dipoles"] = training_mae_dipoles.compute()
        maes["mae_quadrupoles"] = training_mae_quadrupoles.compute()

    training_end = time.time()
    training_time = training_end - training_start
    
    return maes, training_time, last_lr


def log_stats(epoch, last_learning_rate, training_time, validation_time, current_tmae, current_vmae, best_epoch, best_vmae, mae_training, mae_validation):
    # log stats
    print("********************************************************************")
    print(f"Training time (s): {training_time:.3f}")
    print(f"Validation time (s): {validation_time:.3f}")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    # printouts
    print("Training MAE")
    print(f"Weighted Training MAE: {current_tmae:.5f}")
    print(f"MAE Energy [kJ/mol]: {mae_training['mae_energies']:.5f}")
    print(f"MAE QM Gradient [kJ/mol]: {mae_training['mae_qm_gradients']:.5f}")
    print(f"MAE MM Gradient [kJ/mol]: {mae_training['mae_mm_gradients']:.5f}")
    if PARAMETERS['multi_loss']:
        print(f"MAE Dipoles [eA]: {mae_training['mae_dipoles']:.5f}")
        print(f"MAE Quadrupoles [eA2]: {mae_training['mae_quadrupoles']:.5f}")

    print("--------------------------------------------------------------------")

    print("Validation MAE")
    print(f"Weighted Validation MAE: {current_vmae:.5f}")
    print(f"MAE Energy [kJ/mol]: {mae_validation['mae_energies']:.5f}")
    print(f"MAE QM Gradient [kJ/mol]: {mae_validation['mae_qm_gradients']:.5f}")
    print(f"MAE MM Gradient [kJ/mol]: {mae_validation['mae_mm_gradients']:.5f}")
    if PARAMETERS['multi_loss']:
        print(f"MAE Dipoles [eA]: {mae_validation['mae_dipoles']:.5f}")
        print(f"MAE Quadrupoles [eA2]: {mae_validation['mae_quadrupoles']:.5f}")

    print("--------------------------------------------------------------------")

    print(f"LR: ", last_learning_rate)
    print("Best epoch: ", best_epoch)
    print(f"Best validation loss: {best_vmae:.5f}", flush=True)

    # tensorboard (hyperparameters)
    writer.add_scalar("Last LR", last_learning_rate, epoch)

    # tensorboard (training)
    writer.add_scalar("Weighted MAE Train", current_tmae, epoch)
    writer.add_scalar("MAE QM Energies Train", mae_training['mae_energies'], epoch)
    writer.add_scalar("MAE QM Grads Train", mae_training['mae_qm_gradients'], epoch)
    writer.add_scalar("MAE MM Grads Train", mae_training['mae_mm_gradients'], epoch)
    if PARAMETERS['multi_loss']:
        writer.add_scalar("MAE Dipoles Train", mae_training['mae_dipoles'], epoch)
        writer.add_scalar("MAE Quadrupoles Train", mae_training['mae_quadrupoles'], epoch)

    # tensorboard (validation)
    writer.add_scalar("Weighted MAE Val", current_vmae, epoch)
    writer.add_scalar("MAE QM Energies Val", mae_validation['mae_energies'], epoch)
    writer.add_scalar("MAE QM Grads Val", mae_validation['mae_qm_gradients'], epoch)
    writer.add_scalar("MAE MM Grads Val", mae_validation['mae_mm_gradients'], epoch)
    if PARAMETERS['multi_loss']:
        writer.add_scalar("MAE Dipoles Val", mae_validation['mae_dipoles'], epoch)
        writer.add_scalar("MAE Quadrupoles Val", mae_validation['mae_quadrupoles'], epoch)

    writer.flush()

def log_general_stats(model_name, experiment_name, mol_charge, random_seed):
    print(f"Model: {model_name}")
    print(f"Experiment: {experiment_name}")
    print(f"Molecular charge (QM zone): {mol_charge}")
    print(f"Random seed: {random_seed}")

def save_model(model, PARAMETERS):
    if not os.path.exists(PARAMETERS["save_path"]):
        os.makedirs(PARAMETERS["save_path"])
    
    # saving model
    torch.save(model, os.path.join(PARAMETERS["save_path"], f"{PARAMETERS['model_name']}.pt"))
    torch.save(model.state_dict(), os.path.join(PARAMETERS["save_path"], f"{PARAMETERS['model_name']}_state_dict.pth"))
    print(f"Saving model (state_dict) for: {model.device}, {model.dtype}")


def save_parameters(PARAMETERS):
    if not os.path.exists(PARAMETERS["save_path"]):
        os.makedirs(PARAMETERS["save_path"])

    # save parameters
    np.save(os.path.join(PARAMETERS["save_path"], "parameters"), PARAMETERS)
    shutil.copyfile(PARAMETERS["parameters_file"], os.path.join(PARAMETERS["save_path"], "parameters_original.yaml"))
    with open(os.path.join(PARAMETERS["save_path"], "parameters_traced.yaml"), 'w') as outfile:
        PARAMETERS_COPY = copy.deepcopy(PARAMETERS)
        PARAMETERS_COPY["dtype"] = str(PARAMETERS_COPY["dtype"]) # torch.dtype cannot be parsed py PyYaml
        yaml.dump(PARAMETERS_COPY, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AMP-QMMM model.")
    parser.add_argument("-c", "--config", dest="parameters_file", type=str, help="Path to the parameters YAML file.")
    parser.add_argument("-g", "--gpu", type=str, default="0", help="GPU to use for training.")
    args = parser.parse_args()

    # load parameters
    PARAMETERS = load_parameters_file(args.parameters_file)
    PARAMETERS["parameters_file"] = args.parameters_file
    PARAMETERS["model_name"] = f'AMPQMMM_model_{PARAMETERS["system_name"]}_{PARAMETERS["experiment_name"]}_coulombqm_{PARAMETERS["time"]}_{PARAMETERS["random_seed"]}_{PARAMETERS["dtype"]}'

    # set seed
    torch.manual_seed(PARAMETERS["random_seed"])
    np.random.seed(PARAMETERS["random_seed"])
    random.seed(PARAMETERS["random_seed"])

    # set precision
    if PARAMETERS["dtype"] == "float32":
        torch.set_default_dtype(torch.float32)
    elif PARAMETERS["dtype"] == "float64":
        torch.set_default_dtype(torch.float64)
    else:
        print(f"Unsupported dtype: {PARAMETERS['dtype']}")
        sys.exit(1)

    # load training and validation data
    if PARAMETERS["single_system"]:
        training_data = SingleSystemOrcaXtbDataset(
            PARAMETERS["data_path"],
            PARAMETERS["system_name"],
            0,
            PARAMETERS["split_indices"][0],
            PARAMETERS["split_indices"][-1],
            PARAMETERS["dtype"],
            PARAMETERS["delta_qmmm"],
            PARAMETERS["delta_qm"],
            PARAMETERS["multi_loss"],
            False,
        )
        validation_data = SingleSystemOrcaXtbDataset(
            PARAMETERS["data_path"],
            PARAMETERS["system_name"],
            PARAMETERS["split_indices"][0],
            PARAMETERS["split_indices"][1],
            PARAMETERS["split_indices"][-1],
            PARAMETERS["dtype"],
            PARAMETERS["delta_qmmm"],
            PARAMETERS["delta_qm"],
            PARAMETERS["multi_loss"],
            False,
        )
        assert training_data._e0 == validation_data._e0
        assert training_data._e0_idx == validation_data._e0_idx
        if PARAMETERS["delta_qmmm"] or PARAMETERS["delta_qm"]:
            assert training_data._de0 == validation_data._de0
    else:
        training_data = MultiSystemOrcaXtbDataset(
            PARAMETERS["data_path"],
            PARAMETERS["system_name"],
            "training",
            PARAMETERS["dtype"],
            PARAMETERS["delta_qmmm"],
            PARAMETERS["delta_qm"],
            PARAMETERS["multi_loss"],
        )
        validation_data = MultiSystemOrcaXtbDataset(
            PARAMETERS["data_path"],
            PARAMETERS["system_name"],
            "validation",
            PARAMETERS["dtype"],
            PARAMETERS["delta_qmmm"],
            PARAMETERS["delta_qm"],
            PARAMETERS["multi_loss"],
        )

    model = instantiate_model(PARAMETERS, training_data)
    model = set_model_dtype(model, PARAMETERS)
    assert_correct_dtype(model, PARAMETERS)
        
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=PARAMETERS['learning_rate'])
    
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    # scheduler
    decay_rate = np.exp(np.log(PARAMETERS['decay_factor']) / (PARAMETERS["num_epochs"]))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
    print(f"ExponentialLR scheduler with decay rate {decay_rate}")
    
    # instantiate data loaders
    if PARAMETERS["single_system"]:
        training_loader = DataLoader(training_data, batch_size=PARAMETERS["batch_size"], shuffle=True, drop_last=True)
        validation_loader = DataLoader(validation_data, batch_size=PARAMETERS["batch_size"], shuffle=False, drop_last=True)
    else:
        collate_function = def_collate_fn(PARAMETERS["batch_size"])
        training_loader = DataLoader(training_data, batch_size=1, shuffle=True, drop_last=True, collate_fn=collate_function)
        validation_loader = DataLoader(validation_data, batch_size=1, shuffle=False, drop_last=True, collate_fn=collate_function)

    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    PARAMETERS["experiment_name"] = f"{PARAMETERS['model_name']}_{PARAMETERS['n_steps']}_{PARAMETERS['cutoff']}A_POL{PARAMETERS['cutoff_lr']}A_{PARAMETERS['num_epochs']}x{len(training_loader)}x{PARAMETERS['batch_size']}_alpha_{PARAMETERS['alpha']}_beta_{PARAMETERS['beta']}_gamma_{PARAMETERS['gamma']}D{PARAMETERS['delta_qm']}D{PARAMETERS['delta_qmmm']}_M{PARAMETERS['multi_loss']}Q_N{PARAMETERS['n_channels']}FILTER_N{PARAMETERS['n_kernels']}KERNELS"
    PARAMETERS["summary_path"] = os.path.join("summaries", PARAMETERS["experiment_name"])
    PARAMETERS["save_path"] = os.path.join("results", PARAMETERS["experiment_name"])
    log_general_stats(PARAMETERS["model_name"], PARAMETERS["experiment_name"], PARAMETERS["mol_charge"], PARAMETERS["random_seed"])
    print(f"Number of parameters that require gradient: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    if PARAMETERS["device_name"] == "cuda":
        print(f"Training on device: {torch.cuda.get_device_name(model.device)}")
        print(torch.cuda.get_device_properties(model.device))
    else:
        print("Training on CPU.")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    
    writer = SummaryWriter(PARAMETERS["summary_path"])

    # keep track of best epoch w.r.t. validation loss for early stopping
    PARAMETERS["best_epoch"] = -1
    PARAMETERS["best_vmae"] = float("inf")

    for epoch in range(PARAMETERS["num_epochs"]):
        print("********************************************************************")
        print(f"Starting epoch {epoch}...")
        
        # training and validation
        mae_training, times_training, last_learning_rate = train_one_epoch(epoch, model, optimizer, loss_fn, training_loader, PARAMETERS)
        mae_validation, validation_time = evaluate_on_dataset(model, "validation", validation_loader, PARAMETERS)

        # compute MAE (training)
        tmae_potential = (1 - PARAMETERS['alpha']) * mae_training["mae_energies"]
        tmae_qm_gradient = PARAMETERS['alpha'] * mae_training["mae_qm_gradients"]
        tmae_mm_gradient = PARAMETERS['alpha'] * mae_training["mae_mm_gradients"]     
        current_tmae = tmae_potential + tmae_qm_gradient + tmae_mm_gradient
        if PARAMETERS['multi_loss']:
            tmae_dipoles = PARAMETERS['gamma'] * mae_training["mae_dipoles"]  
            tmae_quadrupoles = PARAMETERS['gamma'] * mae_training["mae_quadrupoles"]  
            current_tmae = current_tmae + tmae_dipoles + tmae_quadrupoles

        # compute MAE (validation)
        vmae_potential = (1 - PARAMETERS['alpha']) * mae_validation["mae_energies"]
        vmae_qm_gradient = PARAMETERS['alpha'] * mae_validation["mae_qm_gradients"]
        vmae_mm_gradient = PARAMETERS['alpha'] * mae_validation["mae_mm_gradients"]     
        current_vmae = vmae_potential + vmae_qm_gradient + vmae_mm_gradient
        if PARAMETERS['multi_loss']:
            vmae_dipoles = PARAMETERS['gamma'] * mae_validation["mae_dipoles"]  
            vmae_quadrupoles = PARAMETERS['gamma'] * mae_validation["mae_quadrupoles"]  
            current_vmae = current_vmae + vmae_dipoles + vmae_quadrupoles
        PARAMETERS["epochs_trained"] = epoch + 1 # offset 1
        PARAMETERS["last_tmae"] = current_tmae.item()
        PARAMETERS["last_vmae"] = current_vmae.item()

        # update scheduler
        scheduler.step()
        
        # save model if current validation MAE is lowest (for early stopping)
        if current_vmae < PARAMETERS["best_vmae"]:
            print("********************************************************************")
            print(f"current_vmae ({current_vmae.item():.5f}) < best_vmae ({PARAMETERS['best_vmae']:.5f}), saving model...")
            PARAMETERS["best_vmae"] = current_vmae.item()
            PARAMETERS["best_epoch"] = epoch + 1 # offset
            save_model(model, PARAMETERS)

        # log stats and write tensorboard
        log_stats(epoch, last_learning_rate, times_training, validation_time, current_tmae, current_vmae, PARAMETERS["best_epoch"], PARAMETERS["best_vmae"], mae_training, mae_validation)
        save_parameters(PARAMETERS)

    # assert the model learned something
    assert PARAMETERS["best_epoch"] != -1
    assert PARAMETERS["best_vmae"] < float("inf")
    print("Training finished successfully.")
