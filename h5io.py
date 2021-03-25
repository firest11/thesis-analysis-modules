import os
import numpy as np
import h5py


def get_pvec_indices(h5dat, pvec_tag, pvecs):
    assert len(pvecs[0]) == 3
    try:
        plist = h5dat.attrs[pvec_tag]
    except KeyError:
        print(h5dat.attrs.keys(), pvec_tag)
        h5dat.close()
        raise Exception("pvec not available in h5data_set")
    pinds = []
    for p_v in pvecs:
        for index, p_l in enumerate(plist):
            if all(np.array(p_v) == p_l):
                pinds.append(index)
                break
    return pinds


def get_h5_data_single(h5dat, h5key, pvec_tag, pvecs, ncfg_cap=None):
    def get_data(h5dat, h5key, pinds, ncfg_cap=None):
        # import pdb; pdb.set_trace()
        data = h5dat[h5key][...]
        # data = np.squeeze(data)
        # print(data.shape)
        before = data.shape[0]
        if ncfg_cap is not None:
            data = data[:before-ncfg_cap]
        # after = data.shape[0]
        # print(f'Before {before}: After {after}: Diff {before - after}: Cap {ncfg_cap}')
        if data.ndim == 4:
            data = data[:, 0, :, :]
        if "qpdf" in h5dat.filename and "coarse" in h5dat.filename:
            data = data[..., pinds, :16]
        else:
            data = data[..., pinds, :]
        return data
    pinds = get_pvec_indices(h5dat, pvec_tag, pvecs)
    # print(pinds)
    try:
        data = get_data(h5dat, h5key, pinds, ncfg_cap)
        # data = np.squeeze(data)
        # print(f"Out Data Shape: {data.shape}")
    except Exception as e:
        print(h5dat.filename, h5key)
        h5dat.close()
        raise Exception("ERROR: {0}".format(e))
    return data


def get_h5_data(h5data_set, h5key, pvec_tag, pvecs, caps=None):
    if caps is not None:
        data_set = [
            get_h5_data_single(h5dat, h5key, pvec_tag, pvecs, ncfg_cap=cap)
            for h5dat, cap in zip(h5data_set, caps)
        ]
    else:
        data_set = [get_h5_data_single(h5dat, h5key, pvec_tag, pvecs)
                    for h5dat in h5data_set]
    # ncfg_tot = 0
    # for dat in data_set:
    #     ncfg_tot += dat.shape[0]
    #     print(dat.shape[0], ncfg_tot)
    data_tot = np.vstack(data_set)
    return data_tot


def save_data(h5name, dset_vals):
    fmode = 'r+' if os.path.exists(h5name) else 'w'
    h5f = h5py.File(h5name, fmode)
    print("%s opened/created" % h5name)
    for dkey, dval in dset_vals.items():
        mshape, mtype = dval.shape, dval.dtype
        try:
            dset = h5f[dkey]
            dset[:] = dval
        except KeyError:
            print("Key Error -- Making Dataset")
            dset = h5name.create_dataset(dkey, shape=mshape, dtype=mtype,
                                         fletcher32=True)
            dset[:] = dval
        print("%s[%s] <-- data" % (h5name, dkey))
    h5name.file.flush()
    h5name.close()
    print("%s closed" % h5name)
