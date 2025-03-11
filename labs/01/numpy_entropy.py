#!/usr/bin/env python3
import argparse

import numpy as np

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--data_path", default="numpy_entropy_data_4.txt", type=str, help="Data distribution path.")
parser.add_argument("--model_path", default="numpy_entropy_model_4.txt", type=str, help="Model distribution path.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[float, float, float]:
    # TODO: Load data distribution, each line containing a datapoint -- a string.
    d_count = {}
    t_count = 0
    with open(args.data_path, "r") as data:
        for line in data:
            line = line.rstrip("\n")
            # TODO: Process the line, aggregating data with built-in Python
            # data structures (not NumPy, which is not suitable for incremental
            # addition and string mapping).
            d_count[line] = d_count.get(line, 0) + 1
            t_count += 1

    # TODO: Create a NumPy array containing the data distribution. The
    # NumPy array should contain only data, not any mapping. Alternatively,
    # the NumPy array might be created after loading the model distribution.
    
    # d_prob = {}
    # for k, v in d_count.items(): 
    #     d_prob[k] = v / t_count
    d_prob = {k: v / t_count for k, v in d_count.items()}

    # TODO: Load model distribution, each line `string \t probability`.
    m_prob = {}
    with open(args.model_path, "r") as model:
        for line in model:
            line = line.rstrip("\n")
            # TODO: Process the line, aggregating using Python data structures.
            token, prob = line.split("\t")
            m_prob[token] = float(prob) # need this !!!
            
    # TODO: Create a NumPy array containing the model distribution.
    d_key = np.array(list(d_prob.keys()))
    d_value = np.array(list(d_prob.values()))
    m_value = np.array([m_prob.get(k,0) for k in d_key])

    # TODO: Compute the entropy H(data distribution). You should not use
    # manual for/while cycles, but instead use the fact that most NumPy methods
    # operate on all elements (for example `*` is vector element-wise multiplication).
    entropy = -np.sum(d_value * np.log(d_value))

    # TODO: Compute cross-entropy H(data distribution, model distribution).
    # When some data distribution elements are missing in the model distribution,
    # return `np.inf`.
    if np.any(m_value == 0):
        crossentropy = np.inf
    else:
        crossentropy = -np.sum(d_value * np.log(m_value)) 

    # TODO: Compute KL-divergence D_KL(data distribution, model_distribution),
    # again using `np.inf` when needed.
    kl_divergence = np.sum(d_value * np.log(d_value/m_value))

    # Return the computed values for ReCodEx to validate.
    return entropy, crossentropy, kl_divergence


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    entropy, crossentropy, kl_divergence = main(main_args)
    print("Entropy: {:.2f} nats".format(entropy))
    print("Crossentropy: {:.2f} nats".format(crossentropy))
    print("KL divergence: {:.2f} nats".format(kl_divergence))
