import csv
import itertools
import numpy as np

def write(filename, keys, values):
    assert len(keys) == len(values)

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(keys)

        for vs in itertools.zip_longest(*values):
            writer.writerow(vs)

def read(filename, is_float=True, skip=0):
    keys = []
    keys_cnt = {}
    values = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i < skip: continue
            if i == skip:
                for key in row:
                    if key == '': continue

                    if key not in keys_cnt:
                        keys_cnt[key] = 0
                        keys.append(key)
                    else:
                        keys_cnt[key] = keys_cnt[key] + 1
                        keys.append(key+str(keys_cnt[key]))
                    values.append([])

            else:
                for j, v in enumerate(row):
                    if j < len(keys):
                        if v == '':
                            values[j].append(None)
                        else:
                            if is_float:
                                values[j].append(float(v))
                            else:
                                values[j].append(v)
    data = {}
    for k,v in zip(keys,values):
        data[k] = v

    return data

def detect_change(d,window=10,threshold=0.0001):
    d_init = np.average(d[0:window])
    for i in range(0,len(d),window):
        d_init_c = np.average(d[i:i+window])
        if np.abs(d_init_c-d_init) > threshold:
            break
        d_init = d_init_c
    return i
