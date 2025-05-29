import random
import ot
import time
from datetime import datetime
import math
import numpy as np
import torch
from scipy.stats import truncnorm
import argparse
from tqdm import tqdm

import pyximport

pyximport.install()

from constants import *
from DataStructures.point import Point
from Algorithms.pWasserstein import compute_p_wasserstein
from utils import (
    compute_euclidean_distances,
    compute_l1_distances,
)
from utils import compute_approximation_factors


def sinkhorn(a, b, C): 
    reg = 0.01
    max_iters = 300
    K = torch.exp(-C/reg)
    u = torch.ones_like(a)
    v = torch.ones_like(b)
    for i in range(max_iters):
        u = a / torch.matmul(K,v)
        v = b / torch.matmul(K.T,u)
    return torch.sum(torch.matmul(torch.diag_embed(u), torch.matmul(K, torch.diag_embed(v))) * C)


def generate_points(args):
    """
    Generates the point sets with the corresponding cost matrix
    :param compute_matrix: a function that computes the cost matrix
    """
    n = args.sample_size
    d = args.dim
    if not args.ot:
        masses_a = [1 for _ in range(n)]
        masses_b = [1 for _ in range(n)]
    else:
        delta = 1 / (10 * n)
        masses_a = [0 for _ in range(n)]
        masses_b = [0 for _ in range(n)]
        for _ in range(int(1 / delta)):
            masses_a[random.randint(0, n - 1)] += 1
            masses_b[random.randint(0, n - 1)] += 1
        masses_a = np.array(masses_a) * delta
        masses_b = np.array(masses_b) * delta

    if args.distribution == "Normal_same":
        # Define the parameters
        lower_bound = 0
        upper_bound = 1
        mean = 0.4
        std_dev = 0.1
        
        na = (lower_bound - mean) / std_dev
        nb = (upper_bound - mean) / std_dev
        
        A = [
            Point(list(truncnorm.rvs(na, nb, loc=mean, scale=std_dev, size=d)), i, masses_a[i])
            for i in range(n)
        ]
        B = [
            Point(list(truncnorm.rvs(na, nb, loc=mean, scale=std_dev, size=d)), n + i, masses_b[i])
            for i in range(n)
        ] 
    elif args.distribution == "Normal_different":
        # Define the parameters
        lower_bound = 0
        upper_bound = 1
        mean = 0.3
        std_dev = 0.3
        
        na = (lower_bound - mean) / std_dev
        nb = (upper_bound - mean) / std_dev
        
        A = [
            Point(list(truncnorm.rvs(na, nb, loc=mean, scale=std_dev, size=d)), i, masses_a[i])
            for i in range(n)
        ]
        mean = 0.7
        na = (lower_bound - mean) / std_dev
        nb = (upper_bound - mean) / std_dev
        B = [
            Point(list(truncnorm.rvs(na, nb, loc=mean, scale=std_dev, size=d)), n + i, masses_b[i])
            for i in range(n)
        ] 
    elif args.distribution == "Random_plane":
        while True:
            # Step 1: Randomly generate a basis vector for the plane.
            basis_1 = np.random.rand(d)
            basis_2 = np.random.rand(d)

            # Normalize the basis vectors.
            basis_1 /= np.linalg.norm(basis_1)
            basis_2 /= np.linalg.norm(basis_2)
            
            cosine_similarity = np.dot(basis_1, basis_2)  # Cosine of the angle
            if cosine_similarity <= 1/2:  # Ensure the angle >= 60 degrees
                break
        
        A = [
            Point(list((random.random() / 2) * basis_1 + (random.random() / 2) * basis_2), i, masses_a[i])
            for i in range(n)
        ]
        B = [
            Point(list((random.random() / 2) * basis_1 + (random.random() / 2) * basis_2), n + i, masses_b[i])
            for i in range(n)
        ]
    else:
        A = [
            Point([float(random.random()) for _ in range(d)], i, masses_a[i])
            for i in range(n)
        ]
        B = [
            Point([float(random.random()) for _ in range(d)], n + i, masses_b[i])
            for i in range(n)
        ]

    return A, B, masses_a, masses_b


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='p-Wasserstein Algorithm')
    
    parser.add_argument('--sample_size', type=int, default=10000, help='input sample size')
    parser.add_argument('--p', type=int, default=2, help='parameter p for p-Wasserstein distance')
    parser.add_argument('--dim', type=int, default=10, help='dimension of generated points')
    parser.add_argument('--distribution', choices=["Uniform", "Normal_same", "Normal_different", "Random_plane"], default="Uniform", help='The distribution of the points')
    parser.add_argument('--ot', action='store_true', help='Use random masses on the points')
    
    args = parser.parse_args()

    now = datetime.now()
    # Format the datetime object
    formatted_datetime = now.strftime("%Y-%m-%d-%H-%M")

    filename = (
        "N"
        + str(args.sample_size)
        + "_P"
        + str(args.p)
        + "_D"
        + str(args.dim)
        + "_date"
        + str(formatted_datetime)
        + ".txt"
    )
    if args.distribution == "Uniform":
        filename = "Uniform_" + filename
    elif args.distribution == "Normal_same":
        filename = "Normal_" + filename
    elif args.distribution == "Normal_different":
        filename = "NormalDiff_" + filename
    elif args.distribution == "Random_plane":
        filename = "Plane_" + filename
    if not args.ot:
        filename = "results/result_M_" + filename
    else:
        filename = "results/result_OT_" + filename


    for _ in tqdm(range(5)):
        print("Sample size:", args.sample_size, "p:", args.p, "dim:", args.dim, "distribution:", args.distribution, "matching:", not args.ot)
        
        print("Generating points...")
        A, B, masses_a, masses_b = generate_points(args)
        
        new_C = []
        if args.sample_size <= 20000 and COST_MATRIX:
            new_C = [[-1 for _ in range(args.sample_size)] for _ in range(args.sample_size)]
        
        print("Computing p-Wasserstein distance using our algorithm...")
        time_before = time.time()
        matching_cost, total_count, counter, hungarian, fake_matching_cost, construction_time, hungarian_time, new_C = compute_p_wasserstein(A, B, args.sample_size, args.p, args.dim, args.p, new_C, not args.ot)
        time_after = time.time()

        print("Computing p-Wasserstein distance using EMD...")
        time_cost = time.time()
        if args.sample_size < 60000:
            if IS_L1:
                C = compute_l1_distances(A, B, args.p)
            else:
                C = compute_euclidean_distances(A, B, args.p)
        else:
            C = None
            
        time_cost = time.time() - time_cost

        
        if C is not None:
            time_before_emd = time.time()
            cost = ot.emd2(masses_a, masses_b, C)
            if not args.ot:
                cost = cost / args.sample_size
            real_cost = math.pow(
                cost, 1 / float(args.p)
            )
            time_after_emd = time.time()

            matrix_time = 0
        else:
            real_cost = 1
            real_C = []
            time_before_emd = 0
            time_after_emd = 0
            matrix_time = 0
            
        if C is not None and RUN_SINKHORN: 
            print("Computing p-Wasserstein distance using Sinkhorn...")
            time_sinkhorn = time.time()
            cost = sinkhorn(torch.Tensor(masses_a), torch.Tensor(masses_b), torch.Tensor(C))
            if not args.ot:
                cost /= args.sample_size
            sinkhorn_cost = math.pow(cost, 1 / float(args.p))
            time_sinkhorn = time.time() - time_sinkhorn
        else:
            time_sinkhorn = 0
            sinkhorn_cost = 0
            
        if args.sample_size <= 20000 and COST_MATRIX:
            compute_approximation_factors(C, new_C, args.p, args.dim, formatted_datetime, args.p)
        

        with open(filename, "a") as f:
            f.write(
                "\t".join(
                    [
                        "n:",
                        str(args.sample_size),
                        "p:",
                        str(args.p),
                        "d:",
                        str(args.dim),
                        "m:",
                        str(args.p),
                        "Real cost:",
                        f"{real_cost:.5f}",
                        "Our cost:",
                        f"{matching_cost:.5f}",
                        "Ratio:",
                        f"{matching_cost / real_cost:.5f}",
                        "Our fake cost:",
                        f"{fake_matching_cost:.5f}",
                        "Ratio fake:",
                        f"{fake_matching_cost / real_cost:.5f}",
                        "Time Ours:",
                        f"{time_after - time_before:.2f}",
                        "Time Construction:",
                        f"{construction_time:.2f}",
                        "Time Hungarian:",
                        f"{hungarian_time:.2f}",
                        "Time EMD + costs:",
                        f"{time_cost + time_after_emd - time_before_emd:.2f}",
                        "Time EMD:",
                        f"{time_after_emd - time_before_emd:.2f}",
                        "sinkhorn cost:",
                        f"{sinkhorn_cost:.5f}",
                        "sinkhorn time:",
                        f"{time_sinkhorn:.5f}",
                        "Ratio sinkhorn:",
                        f"{sinkhorn_cost / real_cost:.5f}",
                        "Transported mass:",
                        f"{total_count:.5f}",
                        "Number of iterations:",
                        f"{counter}",
                        "Time cost matrix:",
                        f"{time_cost:.2f}",
                        "Hung node visits:", 
                        str(hungarian.get_visited_nodes()),
                        "Hung edge visits:", 
                        str(hungarian.get_visited_edges()),
                        "DFS node visits:", 
                        str(hungarian.get_dfs_visited_nodes()),
                        "DFS edge visits:", 
                        str(hungarian.get_dfs_visited_edges()),
                        "\n",
                    ]
                )
            )

        del A
        del B
        del C

