import pandas as pd
import numpy as np
import os
import struct


def load_projections_data(proj_path: str):
    with open(proj_path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(0, 0)
        data = f.read(size)

    unpacked = []
    entry_size = 4272
    s = struct.Struct('38i 256s 4x 964i 4x')
    for i in range(len(data) // entry_size):
        unpacked.append([*s.unpack(data[i * entry_size:(i + 1) * entry_size])])
        unpacked[-1][38] = unpacked[-1][38].decode().strip('\0')

    res = pd.DataFrame(unpacked)
    res = res[np.arange(38, 1003)]
    res.columns = np.arange(0, 965)
    return res


def load_data_directory(directory: str,
                        proj_name: str = 'projs.tbl',
                        params_name: str = 'params_mod.txt'):
    proj = load_projections_data(os.path.join(directory, proj_name))[np.arange(641)]
    info = pd.read_csv(os.path.join(directory, params_name), sep=r'\s+')
    return info[['Filename', 'PupX', 'PupY', 'PupR', 'IriX', 'IriY', 'IriR']].\
        set_index('Filename').join(proj.set_index(0))


def load_dataset(root: str,
                 proj_name: str = 'projs.tbl',
                 params_name: str = 'params_mod.txt'):
    dirs = os.listdir(root)
    datas = []
    for dir in dirs:
        datas.append(load_data_directory(os.path.join(root, dir), proj_name, params_name))
    if len(datas) > 0:
        return pd.concat(datas)
