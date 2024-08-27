import os
import shutil
from pathlib import Path
from tqdm import tqdm

# Mapping of instance classes
INSTANCE_CLASS_MAPPING = {
    '3_regular_graph': '3-Regular Graph',
    '4_regular_graph': '4-Regular Graph',
    'geometric': 'Geometric',
    'nearly_complete_bi_partite': 'Nearly Complete BiPartite',
    'power_law_tree': 'Power Law Tree',
    'uniform_random': 'Uniform Random',
    'watts_strogatz_small_world': 'Watts-Strogatz small world'
}

def clean_filename(filename):
    cleaned = filename.lstrip("0123456789_ ")
    cleaned = cleaned.replace(" ", "_")
    cleaned = cleaned.lower()
    return cleaned

def get_instance_class(directory):
    for evolved_name, original_name in INSTANCE_CLASS_MAPPING.items():
        if evolved_name in directory.lower() or original_name.lower() in directory.lower():
            return evolved_name
    return None

def count_files(directory):
    return sum(len(files) for _, _, files in os.walk(directory))

def process_instances(root_dir, output_dir):
    total_files = count_files(os.path.join(root_dir, 'evolved_instances')) + count_files(os.path.join(root_dir, 'instances'))
    
    with tqdm(total=total_files, desc="Overall Progress") as pbar:
        for category in ['evolved_instances', 'instances']:
            category_path = os.path.join(root_dir, category)
            is_evolved = category == 'evolved_instances'
            
            for subdir in os.listdir(category_path):
                subdir_path = os.path.join(category_path, subdir)
                if os.path.isdir(subdir_path):
                    instance_class = get_instance_class(subdir)
                    if instance_class is None:
                        print(f"Warning: Unknown instance class for directory {subdir}")
                        continue
                    
                    files = [f for f in os.listdir(subdir_path) if f.endswith('.graphml')]
                    
                    for filename in tqdm(files, desc=f"Processing {subdir}", leave=False):
                        clean_name = clean_filename(filename)
                        label = 'evolved' if is_evolved else 'original'
                        new_filename = f"{label}_{instance_class}_{clean_name}"
                        
                        output_subdir = os.path.join(output_dir, instance_class)
                        os.makedirs(output_subdir, exist_ok=True)
                        
                        src = os.path.join(subdir_path, filename)
                        dst = os.path.join(output_subdir, new_filename)
                        shutil.copy2(src, dst)
                        
                        pbar.update(1)

if __name__ == "__main__":
    root_dir = "data"
    output_dir = os.path.join(root_dir, "instances-final")
    
    os.makedirs(output_dir, exist_ok=True)
    
    process_instances(root_dir, output_dir)
    print("Processing complete. Instances saved in data/instances-final")