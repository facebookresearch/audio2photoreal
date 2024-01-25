import argparse

import numpy as np
from scipy import linalg


def calculate_diversity(activation: np.ndarray, diversity_times: int = 10_000) -> float:
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]
    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist


def calculate_activation_statistics(
    activations: np.ndarray,
) -> (np.ndarray, np.ndarray):
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def main(args):
    num_samples = 5
    results = np.load(args.results, allow_pickle=True).item()
    pred_reshaped = results["motion"].squeeze().reshape((num_samples, -1, 104, 600))
    gt_reshaped = results["gt"].squeeze().reshape((num_samples, -1, 104, 600))

    # calulate variance across the different samples generated
    cross_sample_var = np.var(pred_reshaped.reshape((num_samples, -1)), axis=0)
    print("cross var", cross_sample_var.mean())

    pred_pose_last = pred_reshaped.transpose((0, 1, 3, 2)).reshape(-1, 104)
    gt_pose_last = gt_reshaped.transpose((0, 1, 3, 2)).reshape(-1, 104)
    # calculate the static and kinematic diversity
    var_g = calculate_diversity(pred_pose_last)
    print("var_g", var_g.mean())
    var_k = np.var(pred_reshaped, axis=-1)
    print("var_k", var_k.mean())

    # calculate the static and kinematic fid
    pred_mu_g, pred_cov_g = calculate_activation_statistics(pred_pose_last)
    gt_mu_g, gt_cov_g = calculate_activation_statistics(gt_pose_last)
    fid_g = calculate_frechet_distance(gt_mu_g, gt_cov_g, pred_mu_g, pred_cov_g)
    print("fid_g", fid_g)
    # reshape for kinematic fid
    pred_motion = pred_reshaped[..., 1:] - pred_reshaped[..., :-1]
    gt_motion = gt_reshaped[..., 1:] - gt_reshaped[..., :-1]
    pred_motion_last = pred_motion.transpose((0, 1, 3, 2)).reshape(-1, 104)
    gt_motion_last = gt_motion.transpose((0, 1, 3, 2)).reshape(-1, 104)
    pred_mu_k, pred_cov_k = calculate_activation_statistics(pred_motion_last)
    gt_mu_k, gt_cov_k = calculate_activation_statistics(gt_motion_last)
    fid_k = calculate_frechet_distance(gt_mu_k, gt_cov_k, pred_mu_k, pred_cov_k)
    print("fid_k", fid_k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    args = parser.parse_args()
    main(args)
