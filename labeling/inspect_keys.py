"""Quick HDF5 structure inspector — run this first to confirm key names."""
import sys, h5py

def print_tree(g, prefix="", max_depth=4, depth=0):
    if depth > max_depth:
        return
    for k in g.keys():
        item = g[k]
        if isinstance(item, h5py.Dataset):
            print(f"{prefix}{k}  shape={item.shape}  dtype={item.dtype}")
        else:
            print(f"{prefix}{k}/")
            print_tree(item, prefix + "  ", max_depth, depth + 1)

path = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\ceti\Documents\05_data\teleop-sorting\avatar\000\episode.hdf5"
with h5py.File(path, "r") as f:
    print(f"Root attrs: {dict(f.attrs)}\n")
    print_tree(f)
