import h5py
import os

def inspect_h5_weights(h5_path):
    print(f"\nInspecting H5 file: {h5_path}")
    if not os.path.exists(h5_path):
        print("ERROR: File not found.")
        return

    try:
        with h5py.File(h5_path, 'r') as f:
            print("\n--- H5 File Structure ---")
            def print_structure(name, obj):
                print(name)
            f.visititems(lambda name, obj: print_structure(name, obj))

            print("\n--- Top-Level Keys ---")
            for key in f.keys():
                print(" ", key)

            print("\n--- Dumping Available Weights ---")
            for layer in f:
                print(f"Layer: {layer}")
                try:
                    for weight_name in f[layer]:
                        data = f[layer][weight_name][()]
                        print(f"  {weight_name}: shape={data.shape}, dtype={data.dtype}")
                except Exception as e:
                    print(f"  Cannot read weights: {e}")

    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    h5_path = r"C:\Users\HP\Downloads\PycharmProjects\hearing_aid_DTLN\DTLN_norm_500h.h5"
    inspect_h5_weights(h5_path)
