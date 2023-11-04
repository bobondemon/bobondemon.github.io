import numpy as np
import torch
import matplotlib.pyplot as plt

from helper import load_pretrained_model_and_save, load_params_and_get_weights_17_45


def plot_weight_hist(weight17, weight45):
    w17min, w17max = np.min(weight17), np.max(weight17)
    w45min, w45max = np.min(weight45), np.max(weight45)
    # plot the weights histogram
    fontsize = 12
    fig, axs = plt.subplots(1, 2)
    axs[0].hist(weight17, bins=100, linewidth=0.5, edgecolor="white", density=True)
    axs[0].set_title("Layer #17;\n(wmin, wmax) = ({:.2}, {:.2})".format(w17min, w17max))
    axs[0].set_xlabel("weight value", fontsize=fontsize)
    axs[0].set_ylabel("probability mass (%)", fontsize=fontsize)
    axs[0].grid(True)
    axs[1].hist(weight45, bins=100, linewidth=0.5, edgecolor="white", density=True)
    axs[1].set_title("Layer #45;\n(wmin, wmax) = ({:.2}, {:.2})".format(w45min, w45max))
    axs[1].set_xlabel("weight value", fontsize=fontsize)
    axs[1].set_ylabel("probability mass (%)", fontsize=fontsize)
    axs[1].grid(True)
    fig.suptitle("Weights histogram in probability mass")
    fig.tight_layout()
    plt.show()


def cal_smax(w, bit_num):
    # note that we assume zero_point is always 0 (symmetric quant.)
    w_max = np.max(np.abs(w))
    return w_max / 2 ** (bit_num - 1)


def theoretical_mse_qerror(w, clipping_scalar, bit_num, bins=500):
    hist, bin_edges = np.histogram(np.abs(w), bins=bins, density=False)
    hist = hist / np.sum(hist)  # turn into probability mass (note that it is different with density)

    clip_start_idx = np.where(np.diff(bin_edges > clipping_scalar))[0]
    clip_start_idx = 0 if len(clip_start_idx) == 0 else clip_start_idx[0]

    J1 = np.sum(hist[:clip_start_idx]) * (clipping_scalar**2 / (3 * 4**bit_num))
    J2 = 0.0
    for idx in range(clip_start_idx, len(hist)):
        prob_x_mass = hist[idx]
        x = (bin_edges[idx + 1] + bin_edges[idx]) / 2
        J2 += (clipping_scalar - x) ** 2 * prob_x_mass

    return J1 + J2


def do_empirical_qerror_scanning(w, smax, zero, bit_num, scalar_num=100, smax_ratio=2.0):
    bit_num = 4
    quant_min, quant_max = -(2 ** (bit_num - 1)), 2 ** (bit_num - 1) - 1
    qerrors = []
    clipping_scalars = np.linspace(1e-8, smax * smax_ratio, scalar_num)
    # for loop for each scalar
    for cs in clipping_scalars:
        w_q = torch.fake_quantize_per_tensor_affine(
            torch.as_tensor(w), 2 * cs / 2**bit_num, zero, quant_min, quant_max
        ).numpy()
        qerrors.append(np.mean((w - w_q) ** 2))

    return qerrors, clipping_scalars


def do_theoretical_qerror_calculation(w, smax, bit_num, scalar_num=100, smax_ratio=1.0):
    qerrors = []
    clipping_scalars = np.linspace(1e-8, smax * smax_ratio, scalar_num)
    # for loop for each scale
    for cs in clipping_scalars:
        qerrors.append(theoretical_mse_qerror(w, cs, bit_num))
    return qerrors, clipping_scalars


def find_opt_by_Newton_method(weights, bit_num, cs_init=0.0, iter_num=10):
    # `cs` stands for `clipping scalar`
    weights_abs = np.abs(weights)
    cs_cur = cs_init
    for itr in range(iter_num):
        indicator_larger = weights_abs > cs_cur
        indicator_smaller = weights_abs <= cs_cur  # should we ignore case with `==0`?
        numerator = np.sum(weights_abs[indicator_larger])
        denominator = np.sum(indicator_smaller) / (3 * 4**bit_num) + np.sum(indicator_larger)
        cs_cur = numerator / denominator
    return cs_cur


def plot_qerror_layer_17_45(
    qerrors17_em,
    clipping_scalars17_em,
    qerrors45_em,
    clipping_scalars45_em,
    qerrors17_th,
    clipping_scalars17_th,
    qerrors45_th,
    clipping_scalars45_th,
    opt_Newton17,
    opt_Newton45,
):
    fontsize = 12
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(clipping_scalars17_em, qerrors17_em, "-b", clipping_scalars17_th, qerrors17_th, "--r")
    axs[0].plot(
        opt_Newton17[0], opt_Newton17[1], marker="o", markersize=7, markeredgecolor="red", markerfacecolor="green"
    )
    axs[0].set_xlabel("clipping scalar", fontsize=fontsize)
    axs[0].set_ylabel("mse qerror", fontsize=fontsize)
    axs[0].set_title("layer #17", fontsize=fontsize)
    axs[0].legend(["empirical", "theoretical", "minimum by Newton"])
    axs[0].grid(True)

    axs[1].plot(clipping_scalars45_em, qerrors45_em, "-b", clipping_scalars45_th, qerrors45_th, "--r")
    axs[1].plot(
        opt_Newton45[0], opt_Newton45[1], marker="o", markersize=7, markeredgecolor="red", markerfacecolor="green"
    )
    axs[1].set_xlabel("clipping scalar", fontsize=fontsize)
    axs[1].set_ylabel("mse qerror", fontsize=fontsize)
    axs[1].set_title("layer #45", fontsize=fontsize)
    axs[1].legend(["empirical", "theoretical", "minimum by Newton"])
    axs[1].grid(True)

    fig.suptitle("MSE Quantization Error")
    fig.tight_layout()
    plt.show()


def run():
    mdl_path = "./model_params/resnet50_model.pkl"
    load_pretrained_model_and_save(mdl_path, resnet_version=50)
    weight17, weight45 = load_params_and_get_weights_17_45(mdl_path)

    plot_weight_hist(weight17, weight45)
    clipping_scalar_num = 200
    smax_ratio = 7.0

    # quantization config
    bit_num = 4
    zero = 0
    smax17, smax45 = cal_smax(weight17, bit_num), cal_smax(weight45, bit_num)

    # do empirical qerror scanning
    qerrors17, clipping_scalars = do_empirical_qerror_scanning(
        weight17, smax17, zero, bit_num, clipping_scalar_num, smax_ratio
    )
    qerrors45, clipping_scalars = do_empirical_qerror_scanning(
        weight45, smax45, zero, bit_num, clipping_scalar_num, smax_ratio
    )

    # do theoretical qerror calculation
    theoretical_qerrors17, theoretical_clipping_scalars = do_theoretical_qerror_calculation(
        weight17, smax17, bit_num, clipping_scalar_num, smax_ratio
    )
    theoretical_qerrors45, theoretical_clipping_scalars = do_theoretical_qerror_calculation(
        weight45, smax45, bit_num, clipping_scalar_num, smax_ratio
    )

    # find minimum with Newton's method
    opt_Newton_cs17 = find_opt_by_Newton_method(weight17, bit_num, cs_init=0.0)
    opt_Newton_mse17 = theoretical_mse_qerror(weight17, opt_Newton_cs17, bit_num)
    opt_Newton_cs45 = find_opt_by_Newton_method(weight45, bit_num, cs_init=0.0)
    opt_Newton_mse45 = theoretical_mse_qerror(weight45, opt_Newton_cs45, bit_num)

    # plot
    plot_qerror_layer_17_45(
        qerrors17,
        clipping_scalars,
        qerrors45,
        clipping_scalars,
        theoretical_qerrors17,
        theoretical_clipping_scalars,
        theoretical_qerrors45,
        theoretical_clipping_scalars,
        [opt_Newton_cs17, opt_Newton_mse17],
        [opt_Newton_cs45, opt_Newton_mse45],
    )


if __name__ == "__main__":
    run()
