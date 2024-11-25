#!/usr/bin/env python

from Util import (
    SingleSystemOrcaXtbDataset,
    MultiSystemOrcaXtbDataset,
    instantiate_model,
    load_state_dict,
    set_model_dtype,
    assert_correct_dtype,
)
import sys
import copy
import os
import torch
import numpy as np


def save_model(model, device_name, PARAMETERS):
    if not os.path.exists(PARAMETERS["save_path"]):
        os.makedirs(PARAMETERS["save_path"])

    model_float32 = copy.deepcopy(model)
    model_float64 = copy.deepcopy(model)

    device = torch.device(device_name)

    # trace & save model (float32)
    model_float32 = model.to(device, dtype=torch.float32)
    model_float32.device = device
    model_float32.dtype = torch.float32
    model_scripted_float32 = torch.jit.optimize_for_inference(torch.jit.script(model_float32.eval()))
    model_scripted_float32.save(os.path.join(PARAMETERS["save_path"], f"model_float32_{device_name}.pt"))
    print(model_float32.device, model_float32.dtype)

    # trace & save model (float64)
    model_float64 = model.to(device, dtype=torch.float64)
    model_float64.device = device
    model_float64.dtype = torch.float64
    model_scripted_float64 = torch.jit.optimize_for_inference(torch.jit.script(model_float64.eval()))
    model_scripted_float64.save(os.path.join(PARAMETERS["save_path"], f"model_float64_{device_name}.pt"))
    print(model_float64.device, model_float64.dtype)


if __name__ == "__main__":
    usage = f"{sys.argv[0]} results_folder"
    if len(sys.argv) != 2:
        print(usage)
        sys.exit(1)

    # load parameters
    PARAMETERS =  np.load(os.path.join(sys.argv[1], "parameters.npy"), allow_pickle=True).item()

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

    model = instantiate_model(PARAMETERS, training_data)
    model = set_model_dtype(model, PARAMETERS)
    model = load_state_dict(model, PARAMETERS)
    assert_correct_dtype(model, PARAMETERS)

    save_model(model, "cpu", PARAMETERS)
    save_model(model, "cuda", PARAMETERS)
