import pickle
from pathlib import Path
import torchvision.models as models


resnets = {
    18: models.resnet18,
    34: models.resnet34,
    50: models.resnet50,
    101: models.resnet101,
    152: models.resnet152,
}


def load_pretrained_model_and_save(mdl_path_to_save, resnet_version=50):
    mdl_path_to_save = Path(mdl_path_to_save)
    if not mdl_path_to_save.exists():
        Path.mkdir(mdl_path_to_save.parent, parents=True)
        model = resnets[resnet_version](pretrained=True)
        param_dict = dict(model.named_parameters())
        names = list(param_dict.keys())
        param_dict = {"names": names, "params": [p.clone().detach().numpy() for p in param_dict.values()]}
        with open(mdl_path_to_save, "wb") as f:
            pickle.dump(param_dict, f)


def load_params(mdl_path):
    with open(mdl_path, "rb") as f:
        param_dict = pickle.load(f)
    print(param_dict.keys())

    # layer 1
    assert param_dict["names"][0] == "conv1.weight"
    names = [param_dict["names"][0]]
    weights = [param_dict["params"][0]]
    # layer 2~49
    for n, w in zip(param_dict["names"], param_dict["params"]):
        if "layer" in n and "conv" in n:
            names.append(n)
            weights.append(w)
    # # layer 50
    assert param_dict["names"][-2] == "fc.weight"
    names.append(param_dict["names"][-2])
    weights.append(param_dict["params"][-2])

    print(f"len(names)={len(names)}; len(weights)={len(weights)}")
    return names, weights


def get_layer_weight_by_index(weights, layer_idx):
    assert layer_idx >= 0 and layer_idx <= 49, "[Error]: invalid layer index (sould be in range 0~49)"
    return weights[layer_idx]


def load_params_and_get_weights_17_45(mdl_path):
    names, weights = load_params(mdl_path)
    # layer order can ref. https://zhuanlan.zhihu.com/p/442427189
    # weight #17 and #45 are corresponding to 'layer2.2.conv1.weight' and 'layer4.1.conv2.weight'
    assert names[16] == "layer2.2.conv1.weight" and names[44] == "layer4.1.conv2.weight"
    weight17 = get_layer_weight_by_index(weights, 16)  # (128, 512, 1, 1)
    weight45 = get_layer_weight_by_index(weights, 44)  # (512, 512, 3, 3)
    weight17, weight45 = weight17.reshape(-1), weight45.reshape(-1)
    return weight17, weight45
