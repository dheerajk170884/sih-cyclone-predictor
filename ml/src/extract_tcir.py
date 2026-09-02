import os
import h5py
import numpy as np
from PIL import Image

def process_tcir_to_folders(h5_file_path, output_base_dir):
    print(f"Loading TCIR dataset from: {h5_file_path}")
    
    if not os.path.exists(h5_file_path):
        raise FileNotFoundError(f"File not found at {h5_file_path}.")

    # Decode BlockManager dimensions correctly
    with h5py.File(h5_file_path, 'r') as hf:
        info_grp = hf['info']
        
        b0_items = [item.decode('utf-8') if isinstance(item, bytes) else str(item) for item in info_grp['block0_items'][:]]
        b1_items = [item.decode('utf-8') if isinstance(item, bytes) else str(item) for item in info_grp['block1_items'][:]]
        
        if 'Vmax' in b0_items:
            v_idx = b0_items.index('Vmax')
            values = info_grp['block0_values'][:]
        elif 'Vmax' in b1_items:
            v_idx = b1_items.index('Vmax')
            values = info_grp['block1_values'][:]
        else:
            raise KeyError(f"Vmax column not found. Found columns: {b0_items + b1_items}")

        # Ensure vmax_list matches dataset frame length (N)
        if values.shape[0] == len(b0_items) or values.shape[0] == len(b1_items):
            vmax_list = values[v_idx, :]
        else:
            vmax_list = values[:, v_idx]

        matrix = hf['matrix'][:]  # Shape: (N, 201, 201, 4)

    total_frames = len(vmax_list)
    print(f"Total satellite frames found: {total_frames}")

    # Directories setup
    data_dir = os.path.join(output_base_dir, "data")
    identifier_dir = os.path.join(data_dir, "Identifier")
    
    os.makedirs(os.path.join(identifier_dir, "non_cyclone"), exist_ok=True)
    os.makedirs(os.path.join(identifier_dir, "Cyclone"), exist_ok=True)

    imd_classes = {
        "Depression": (17, 27),
        "Deep Depression": (28, 33),
        "Cyclonic Storm": (34, 47),
        "Severe Cyclonic Storm": (48, 63),
        "Very Severe Cyclonic Storm": (64, 119),
        "Super Cyclonic Storm": (120, 300)
    }

    for class_name in imd_classes.keys():
        os.makedirs(os.path.join(data_dir, class_name), exist_ok=True)

    print("Extracting IR1 channels and organizing images...")

    for idx in range(total_frames):
        vmax = float(vmax_list[idx])
        
        ir_image = matrix[idx, :, :, 0]
        ir_image = np.nan_to_num(ir_image)
        
        min_val, max_val = np.min(ir_image), np.max(ir_image)
        if max_val - min_val > 0:
            norm_img = ((ir_image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            norm_img = np.zeros_like(ir_image, dtype=np.uint8)

        img_pil = Image.fromarray(norm_img).convert("RGB")
        img_name = f"tcir_{idx}.png"

        # Separate into Binary Identifier dataset
        if vmax < 17:
            img_pil.save(os.path.join(identifier_dir, "non_cyclone", img_name))
        else:
            img_pil.save(os.path.join(identifier_dir, "Cyclone", img_name))

            # Separate into Multi-class Intensity dataset
            for class_name, (low_kt, high_kt) in imd_classes.items():
                if low_kt <= vmax <= high_kt:
                    img_pil.save(os.path.join(data_dir, class_name, img_name))
                    break

        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1}/{total_frames} images...")

    print("Extraction completed successfully!")

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    TCIR_FILE_PATH = os.path.join(base_dir, "../data/TCIR-CPAC_IO_SH.h5")
    OUTPUT_BASE = os.path.join(base_dir, "..")
    
    process_tcir_to_folders(TCIR_FILE_PATH, OUTPUT_BASE)