import argparse
import datetime
import os
from pathlib import Path

import keras
import numpy as np
import pytz
from einops import rearrange
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = str(-1)  # CPU


def convert_deconv(weight):
    if weight.ndim == 5:
        weight = rearrange(weight, "Z Y X IN OUT -> IN OUT (Z Y X)")
    elif weight.ndim == 4:
        weight = rearrange(weight, "Y X IN OUT -> IN OUT (Y X)")
    else:
        raise NotImplementedError(weight.shape)
    weight = weight[:, :, ::-1]
    return weight


def convert_conv(weight):
    if weight.ndim == 5:
        weight = rearrange(weight, "Z Y X IN OUT -> OUT IN (Z Y X)")
    elif weight.ndim == 4:
        weight = rearrange(weight, "Y X IN OUT -> OUT IN (Y X)")
    else:
        raise NotImplementedError(weight.shape)
    return weight


def format_tag(tag):
    tag = tag.split("/")
    tag = "_".join(tag)
    return tag


def load_checkpoint(checkpoint_path):
    model = keras.models.load_model(checkpoint_path, safe_mode=False)
    step = model.optimizer.iterations.numpy()
    return model.variables, step


def dump2dict(key, param, func, variables_dict, param_dict, used_key_list):
    param_dict[format_tag(key)] = [func(param), param.shape]
    used_key_list.append(key)

    bias_key = key[:-6] + "bias"  # remove kernel and add bias
    if bias_key in variables_dict:
        bias = variables_dict[bias_key]
        param_dict[format_tag(bias_key)] = [bias, bias.shape]
        used_key_list.append(bias_key)


def get_param_dict(variables, skip_list):
    variables_dict = {}
    for variable in variables:
        variables_dict[variable.path] = variable.numpy()

    param_dict = {}
    used_key_list = []

    def _wrapper(key, param, func):
        dump2dict(key, param, func, variables_dict, param_dict, used_key_list)

    for key, param in variables_dict.items():
        if True in [i in key for i in skip_list]:
            print("skipping", key)
            continue
        if "gamma" in key:  # batch norm
            scale_key = key
            offset_key = key[:-5] + "beta"
            moving_mean_key = key[:-5] + "moving_mean"
            moving_variance_key = key[:-5] + "moving_variance"

            scale = param
            offset = variables_dict[offset_key]
            moving_mean = variables_dict[moving_mean_key]
            moving_variance = variables_dict[moving_variance_key]

            bn_param = np.concatenate([moving_mean, moving_variance, offset, scale])
            param_dict[format_tag(key)[:-6]] = [bn_param, bn_param.shape]
            used_key_list += [
                scale_key,
                offset_key,
                moving_mean_key,
                moving_variance_key,
            ]
        elif "transposeconv" in key and param.ndim in [4, 5]:  # 2D/3D transposed conv
            _wrapper(key, param, convert_deconv)
        elif "conv" in key and param.ndim in [4, 5]:  # 2D/3D conv
            _wrapper(key, param, convert_conv)
        elif "fc" in key and param.ndim == 2:  # fc
            _wrapper(key, param, lambda x: x)
    for key in variables_dict.keys():
        if key not in used_key_list:
            print(f"Unused key: {key}")
    return param_dict


def save(text, path):
    with open(path, "w") as f:
        f.write(text)


def export_h(
    param_dict,
    param_name,
    namespace,
    save_path,
    with_extern,
    just_header,
    checkpoint_info,
):
    output = checkpoint_info
    output += f"namespace {namespace} {{\n"
    output += f"\tnamespace {param_name} {{\n"
    if just_header:
        output += "\t\tnamespace Parameter {\n"
    else:
        output += "\t\tstruct Parameter {\n"
    for key, (prm, shape) in tqdm(param_dict.items()):
        if just_header:
            prm = ",".join(map(lambda x: str(x) + "f", prm.flatten()))
            output += f"\t\t\tconst float {key}[] = {{{prm}}}; // {shape}\n\n"
        else:
            output += f"\t\t\tconst float {key}[{np.prod(shape)}]; // {shape}\n\n"
    output += "\t\t};\n"
    if with_extern:
        output += "\t\textern const float g_Parameter[];\n"
    output += "\t}\n"
    output += "}\n"
    save(output, save_path)


def export_cpp(param_dict, param_name, namespace, save_path, checkpoint_info):
    output = checkpoint_info
    output += f'#include "{namespace}{param_name}Params.h"\n\n'
    output += f"namespace {namespace} {{\n"
    output += f"\tnamespace {param_name} {{\n"
    output += "\t\t//array for reinterpret_cast struct Parameter\n"
    output += "\t\tconst float g_Parameter[] = {\n"
    for key, (prm, shape) in tqdm(param_dict.items()):
        output += f"\t\t\t//const float {key}[{np.prod(shape)}];\n"
        output += "\t\t\t"
        output += ",".join(map(lambda x: str(x) + "f", prm.flatten()))
        output += ",\n"
    output += "\t\t};\n"
    output += "\t}\n"
    output += "}\n"
    save(output, save_path)


def export_binary(param_dict, save_path, checkpoint_info):
    with open(save_path, "wb") as f:
        for prm, shape in tqdm(param_dict.values()):
            assert prm.dtype == np.float32
            f.write(prm.flatten().tobytes())
        f.write(checkpoint_info.encode())


def export(
    checkpoint_path, param_name, namespace, exp_name, binary, just_header, skip_list
):
    checkpoint_path = Path(checkpoint_path)

    print("loading checkpoint")
    params, step = load_checkpoint(checkpoint_path)
    print(f"Step: {step}")
    japan_timezone = pytz.timezone("Asia/Tokyo")
    timestamp = datetime.datetime.now(japan_timezone)
    timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    checkpoint_name = checkpoint_path.name
    checkpoint_info = (
        f"/*\n{exp_name}_{checkpoint_name}_step{step}\n{timestamp_str}\n*/\n\n"
    )

    param_dict = get_param_dict(params, skip_list)

    save_h_path = checkpoint_path.parent / f"{namespace}{param_name}Params.h"

    with_extern = False if binary else True
    export_h(
        param_dict,
        param_name,
        namespace,
        save_h_path,
        with_extern,
        just_header,
        checkpoint_info,
    )
    if just_header:
        return None
    if binary:
        save_binary_path = save_h_path.with_suffix(".dat")
        export_binary(param_dict, save_binary_path, checkpoint_info[3:-5])
    else:
        save_cpp_path = save_h_path.with_suffix(".cpp")
        export_cpp(param_dict, param_name, namespace, save_cpp_path, checkpoint_info)


def get_param_num(h_path):
    param_num = 0
    with open(h_path, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if "[" not in line:
                continue
            line = line.split("[")[1]
            line = line.split("]")[0]
            if len(line) == 0:
                continue
            param_num += int(line)
    return param_num


def decode_param_info(dat_path, param_num):
    with open(dat_path, "rb") as f:
        f.seek(param_num * 4)
        binary_data = f.read()
        decoded_string = binary_data.decode()
    return decoded_string


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=Path)
    parser.add_argument("--param_name", type=str, default="CNN")
    parser.add_argument("--namespace", type=str, default="ProstateSegmentation")
    parser.add_argument("--binary", action="store_true", help=" (default false)")
    parser.add_argument("--just_header", action="store_true", help=" (default false)")
    parser.add_argument(
        "--skip_list", nargs="+", type=str, default=[], help="skip name list"
    )
    args = parser.parse_args()

    exp_name = [i for i in str(args.checkpoint_path).split("/") if "exp" in i][0]

    export(
        args.checkpoint_path,
        args.param_name,
        args.namespace,
        exp_name,
        args.binary,
        args.just_header,
        args.skip_list,
    )
