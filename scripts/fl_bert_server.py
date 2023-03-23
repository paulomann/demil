from typing import Dict, List, Optional, OrderedDict, Tuple, Union
import flwr as fl
import pickle
import argparse
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification

import centralized_eval

DEFAULT_SERVER_ADDRESS = "[::]:8080"

def get_args():
    parser = argparse.ArgumentParser(description="Testando criar o servidor para o TensorFlow no CIFAR10 automaticamente")
    parser.add_argument(
        "--server_address", type=str,
        default=DEFAULT_SERVER_ADDRESS,
        help=f"gRPC server address (default: {DEFAULT_SERVER_ADDRESS})",
    )
    parser.add_argument(
        "--rounds", type=int, required=True,
        help="Number of rounds of federated learning",
    )
    parser.add_argument(
        "--sample_fraction", type=float, required=True,
        default=1.0,
        help="Fraction of available clients used for fit/evaluate (default: 1.0)",
    )
    parser.add_argument(
        "--min_sample_size", type=int, required=True,
        default=2,
        help="Minimum number of clients used for fit/evaluate (default: 2)",
    )
    parser.add_argument(
        "--min_num_clients", type=int, required=True,
        default=2,
        help="Minimum number of available clients required for sampling (default: 2)",
    )
    parser.add_argument(
        "--log_host", type=str,
        help="Logserver address (no default)",
    )
    parser.add_argument(
        "--gpu", type=int , required=True,
        default=0,
        help="Wich gpu to put the centralized model"
    )
    args = parser.parse_args()
    return args

def aggregate_metrics(eval_metrics):
    print("====EVAL_METRICS: ", eval_metrics)
    precision, recall, fscore, denominator = 0, 0, 0, 0
    for info in eval_metrics:
        size, metrics = info
        precision += metrics["precision"] * size
        recall += metrics["recall"] * size
        fscore += metrics["fscore"] * size
        denominator += size
    
    return {"precision": precision/denominator, "recall": recall/denominator, "fscore": fscore/denominator}            


def main() -> None:
    args = get_args()
    net = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(args.gpu)

    # Define strategy

    class SaveModelStrategy(fl.server.strategy.FedAvg):
        def aggregate_fit(
            self,
            server_round: int,
            results,
            failures,
        ):
            """Aggregate model weights using weighted average and store checkpoint"""

            # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
            aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

            if aggregated_parameters is not None:
                print(f"Saving round {server_round} aggregated_parameters...")

                # Convert `Parameters` to `List[np.ndarray]`
                aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_weights(aggregated_parameters)

                # Convert `List[np.ndarray]` to PyTorch`state_dict`
                params_dict = zip(net.state_dict().keys(), aggregated_ndarrays)
                state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                net.load_state_dict(state_dict, strict=True)

                # Save the model
                torch.save(net.state_dict(), f"model_round_{server_round}.pth")

            return aggregated_parameters, aggregated_metrics
    
    strategy = SaveModelStrategy(
        fraction_fit=args.sample_fraction,
        fraction_eval=args.sample_fraction,
        min_fit_clients=args.min_sample_size,
        min_eval_clients=args.min_sample_size,
        min_available_clients=args.min_num_clients,
        evaluate_metrics_aggregation_fn=aggregate_metrics
    )
    
    '''
    with open('weights.bin', 'rb') as fp:
        initial_weights = pickle.load(fp)
    initial_parameters = fl.common.weights_to_parameters(initial_weights)
    strategy = fl.server.strategy.FedYogi(
        fraction_fit=args.sample_fraction,
        fraction_eval=args.sample_fraction,
        min_fit_clients=args.min_sample_size,
        min_eval_clients=args.min_sample_size,
        min_available_clients=args.min_num_clients,
        evaluate_metrics_aggregation_fn=aggregate_metrics,
        initial_parameters=initial_parameters
    )
    '''

    # Start Flower server for three rounds of federated learning
    fl.server.start_server(
        server_address=args.server_address,
        config={"num_rounds": args.rounds},
        strategy=strategy,
        grpc_max_message_length=1543194403
    )

    centralized_eval.final_evaluate(net, args.gpu)


if __name__ == "__main__":
    main()