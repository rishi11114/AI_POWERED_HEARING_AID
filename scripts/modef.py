import h5py

MODEL_PATH = "my_model/model.h5"

with h5py.File(MODEL_PATH, "r") as f:
    print("[INFO] Layer Groups in H5 file:")
    for name in f.keys():
        print(name)
    print("\n[INFO] Detailed layer info:")
    for layer in f.keys():
        if isinstance(f[layer], h5py.Group):
            print(f"\nLayer: {layer}")
            for sub in f[layer].keys():
                print(f"  {sub} -> {list(f[layer][sub].keys()) if isinstance(f[layer][sub], h5py.Group) else 'N/A'}")
