import argparse
import yaml
import os

def main():
    parser = argparse.ArgumentParser(description="Generate a YAML file with specified configuration for minimal models.")
    
    parser.add_argument("system_name", help="System name.")
    parser.add_argument("experiment_name", help="Name of the experiment.")
    parser.add_argument("--single_system", action=argparse.BooleanOptionalAction, required=True, help="Train on one or multiple different molecules.")
    parser.add_argument("--split_indices", nargs=4, type=int, help="Split indices for train, validation, test sets. Required for training single systems.")
    parser.add_argument("--random_seed", type=int, default=1301985, help="Seed for random number generation.")
    parser.add_argument("--device_name", default="cuda", help="Device name.")
    parser.add_argument("--model_architecture", default="minimal", help="Model architecture.")
    parser.add_argument("--scheduler", default="ExponentialLR", help="Scheduler type.")
    parser.add_argument("--data_path", default="data", help="Path to data.")
    parser.add_argument("--alpha", type=float, default=0.99, help="Weighting loss function (alpha * RMSE_gradient + (1-alpha) * RMSE_energy).")
    parser.add_argument("--aniso_esp", type=bool, default=True, help="Electrostatic interaction with multipoles between MM and QM particles.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--beta", type=int, default=100, help="Weighting factor MM gradients.")
    parser.add_argument("--gamma", type=int, default=100, help="Weighting factor multipole loss.")
    parser.add_argument("--cutoff", type=float, default=5.0, help="Cutoff QM graph [A].")
    parser.add_argument("--cutoff_lr", type=float, default=9.0, help="Cutoff MM-QM polarization [A].")
    parser.add_argument("--cutoff_esp", type=float, default=500.0, help="Cutoff QM-MM electrostatic interaction [A] ('include everything').")
    parser.add_argument("--decay_factor", type=float, default=0.02, help="Decay factor learning rate.")
    parser.add_argument("--delta_qmmm", type=bool, default=False, help="Use delta learning with electrostatic embedding (QM zone + MM charges).")
    parser.add_argument("--delta_qm", type=bool, default=False, help="Use delta learning without electrostatic embedding (only QM zone).")
    parser.add_argument("--dtype", default="float32", help="DTYPE.")
    parser.add_argument("--learning_rate", type=float, default=0.0005, help="Initial learning rate.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for gradient clipping.")
    parser.add_argument("--multi_loss", type=bool, default=True, help="Include multipole loss term.")
    parser.add_argument("--n_steps", type=int, default=2, help="Number of message passing steps.")
    parser.add_argument("--node_size", type=int, default=128, help="Size of feature vector.")
    parser.add_argument("--num_epochs", type=int, default=128, help="Number of training epochs.")
    parser.add_argument("--n_channels", type=int, default=32, help="Number of multipoles per atom.")
    parser.add_argument("--n_kernels", type=int, default=8, help="Number of Bessel function used to encode the distance between QM-QM particles.")
    parser.add_argument("--n_kernels_qmmm", type=int, default=8, help="Number of Bessel function used to encode the distance between QM-MM particles.")
    parser.add_argument("--order", type=int, default=2, help="Multipole order (only order 2 is implemented).")
    parser.add_argument("--mol_charge", type=float, default=0.0, help="Charge of the system (QM zone).")

    args = parser.parse_args()

    # Build the configuration dictionary
    config = {
        "system_name": args.system_name,
        "experiment_name": args.experiment_name,
        "single_system": args.single_system,
        "random_seed": args.random_seed,
        "device_name": args.device_name,
        "model_architecture": args.model_architecture,
        "scheduler": args.scheduler,
        "data_path": args.data_path,
        "alpha": args.alpha,
        "aniso_esp": args.aniso_esp,
        "batch_size": args.batch_size,
        "beta": args.beta,
        "gamma": args.gamma,
        "cutoff": args.cutoff,
        "cutoff_lr": args.cutoff_lr,
        "cutoff_esp": args.cutoff_esp,
        "decay_factor": args.decay_factor,
        "delta_qmmm": args.delta_qmmm,
        "delta_qm": args.delta_qm,
        "dtype": args.dtype,
        "learning_rate": args.learning_rate,
        "max_grad_norm": args.max_grad_norm,
        "multi_loss": args.multi_loss,
        "n_steps": args.n_steps,
        "node_size": args.node_size,
        "num_epochs": args.num_epochs,
        "n_channels": args.n_channels,
        "n_kernels": args.n_kernels,
        "n_kernels_qmmm": args.n_kernels_qmmm,
        "order": args.order,
        "mol_charge": args.mol_charge,
    }

    if args.single_system:
        assert args.split_indices is not None, "--split_indices is required when single_system is True"
        config["split_indices"] = args.split_indices

    # Write to the YAML file
    filename = os.path.join("../inputs", f"{config['system_name']}_{config['experiment_name']}.yaml")
    with open(filename, 'w') as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False)

if __name__ == "__main__":
    main()
