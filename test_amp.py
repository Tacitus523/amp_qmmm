#!/usr/bin/env python

from Util import (
    SingleSystemOrcaXtbDataset,
    MultiSystemOrcaXtbDataset,
    instantiate_model,
    load_state_dict,
    evaluate_on_dataset,
    set_model_dtype,
    assert_correct_dtype,
)
import sys
import os
from torch.utils.data import DataLoader
import numpy as np


def log_general_stats(model_name, experiment_name, best_epoch, epochs_trained, epochs_scheduled, last_tmae, last_vmae):
    print(f"Model: {model_name}")
    print(f"Experiment: {experiment_name}")
    print(f"Best epoch: {best_epoch}")
    print(f"Epochs trained: {epochs_trained}")
    print(f"Epochs scheduled: {epochs_scheduled}")
    print(f"Last training MAE: {last_tmae}")
    print(f"Last validation MAE: {last_vmae}")

def log_stats(stage_name, maes, evaluation_time):
    print(f"Evaluation time ({stage}) (s): {evaluation_time:.3f}")
    print(f"{stage_name}: Average {stage_name} MAE:")
    print(f"{stage_name}: MAE Energy [kJ/mol]: {maes['mae_energies']:.5f}")
    print(f"{stage_name}: MAE QM Gradient [kJ/mol]: {maes['mae_qm_gradients']:.5f}")
    print(f"{stage_name}: MAE MM Gradient [kJ/mol]: {maes['mae_mm_gradients']:.5f}")
    if PARAMETERS['multi_loss']:
        print(f"{stage_name}: MAE Dipoles [eA]: {maes['mae_dipoles']:.5f}")
        print(f"{stage_name}: MAE Quadrupoles [eA2]: {maes['mae_quadrupoles']:.5f}")

if __name__ == "__main__":
    usage = f"{sys.argv[0]} results_folder"
    if len(sys.argv) != 2:
        print(usage)
        sys.exit(1)

    # load parameters
    PARAMETERS =  np.load(os.path.join(sys.argv[1], "parameters.npy"), allow_pickle=True).item()
    stages = list()
    datasets = dict()

    if PARAMETERS["single_system"]:
        # load training and validation data
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
        datasets["training"] = training_data
        stages.append("training")
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
        datasets["validation"] = validation_data
        stages.append("validation")
        # load test data
        if PARAMETERS["split_indices"][2] == PARAMETERS["split_indices"][3]:
            print("Have 1 test set")
            # one test set
            test_data = SingleSystemOrcaXtbDataset(
                PARAMETERS["data_path"],
                PARAMETERS["system_name"],
                PARAMETERS["split_indices"][1],
                None,
                PARAMETERS["split_indices"][-1],
                PARAMETERS["dtype"],
                PARAMETERS["delta_qmmm"],
                PARAMETERS["delta_qm"],
                PARAMETERS["multi_loss"],
                False,
            )
            datasets["test"] = test_data
            stages.append("test")
        else:
            print("Have 2 test sets")
            # two test sets
            test_data_first = SingleSystemOrcaXtbDataset(
                PARAMETERS["data_path"],
                PARAMETERS["system_name"],
                PARAMETERS["split_indices"][1],
                PARAMETERS["split_indices"][2],
                PARAMETERS["split_indices"][-1],
                PARAMETERS["dtype"],
                PARAMETERS["delta_qmmm"],
                PARAMETERS["delta_qm"],
                PARAMETERS["multi_loss"],
                False,
            )
            datasets["test_inseq"] = test_data_first
            stages.append("test_inseq")
            test_data_second = SingleSystemOrcaXtbDataset(
                PARAMETERS["data_path"],
                PARAMETERS["system_name"],
                PARAMETERS["split_indices"][2],
                None,
                PARAMETERS["split_indices"][-1],
                PARAMETERS["dtype"],
                PARAMETERS["delta_qmmm"],
                PARAMETERS["delta_qm"],
                PARAMETERS["multi_loss"],
                False,
            )
            datasets["test_outseq"] = test_data_second
            stages.append("test_outseq")
    
        # all should have identical e0 and e0_idx
        for dataset_1 in datasets.values():
            for dataset_2 in datasets.values():
                assert dataset_1._e0.item() == PARAMETERS["E0"]
                assert dataset_1._e0_idx.item() == PARAMETERS["E0_IDX"]
                assert dataset_1._e0.item() == dataset_2._e0.item()
                assert dataset_1._e0_idx.item() == dataset_2._e0_idx.item()
                if PARAMETERS["delta_qmmm"] or PARAMETERS["delta_qm"]:
                    assert dataset_1._de0.item() == dataset_2._de0.item()

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
        test_data = MultiSystemOrcaXtbDataset(
            PARAMETERS["data_path"],
            PARAMETERS["system_name"],
            "test",
            PARAMETERS["dtype"],
            PARAMETERS["delta_qmmm"],
            PARAMETERS["delta_qm"],
            PARAMETERS["multi_loss"],
        )
        datasets["training"] = training_data
        datasets["validation"] = validation_data
        datasets["test"] = test_data
        stages.append("training")
        stages.append("validation")
        stages.append("test")

    model = instantiate_model(PARAMETERS, datasets["training"])
    model = set_model_dtype(model, PARAMETERS)
    model = load_state_dict(model, PARAMETERS)
    assert_correct_dtype(model, PARAMETERS)

    print(f"Stages are: {stages}")

    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    log_general_stats(PARAMETERS["model_name"], PARAMETERS["experiment_name"], PARAMETERS["best_epoch"], PARAMETERS["epochs_trained"], PARAMETERS["num_epochs"], PARAMETERS["last_tmae"], PARAMETERS["last_vmae"])
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    # evaluate all datasets
    for stage in stages:
        if PARAMETERS["single_system"]:
            dataloader = DataLoader(datasets[stage], batch_size=PARAMETERS["batch_size"], shuffle=False, drop_last=True)
        else:
            dataloader = DataLoader(datasets[stage], batch_size=1, shuffle=False, drop_last=True, collate_fn=lambda x: x[0])
        maes, evaluation_time = evaluate_on_dataset(model, stage, dataloader, PARAMETERS)
        log_stats(stage, maes, evaluation_time)
        print("--------------------------------------------------------------------------------")

    print("Testing finished successfully.")