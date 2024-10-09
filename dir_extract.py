
import os
import shutil

# Path ke direktori dataset Helen
helen_dir =  "./300W-LP/300W_LP/HELEN"
# Daftar semua file gambar dalam direktori
all_images = os.listdir(helen_dir)

# Tentukan ukuran subset
subset_size = 30000

# Pembagian dataset
for i in range(0, len(all_images), subset_size):
    # Buat direktori baru untuk subset
    subset_dir = f'subset_{i // subset_size + 1}'
    os.makedirs(subset_dir, exist_ok=True)
    
    # Pindahkan file gambar dan file .mat ke direktori subset
    for image in all_images[i:i + subset_size]:
        # Salin gambar
        image_path = os.path.join(helen_dir, image)
        shutil.copy(image_path, os.path.join(subset_dir, image))
        
        # Salin file .mat yang sesuai
        mat_file = image.replace('.jpg', '.mat')  # Pastikan formatnya sesuai
        mat_file_path = os.path.join(helen_dir, mat_file)
        if os.path.exists(mat_file_path):
            shutil.copy(mat_file_path, os.path.join(subset_dir, mat_file))

# Untuk sisa gambar (misalnya 15.000 gambar)
# Buat direktori untuk sisa
remainder_dir = 'subset_remainder'
os.makedirs(remainder_dir, exist_ok=True)

for image in all_images[2 * subset_size:]:
    # Salin gambar
    image_path = os.path.join(helen_dir, image)
    shutil.copy(image_path, os.path.join(remainder_dir, image))
    
    # Salin file .mat yang sesuai
    mat_file = image.replace('.jpg', '.mat')  # Pastikan formatnya sesuai
    mat_file_path = os.path.join(helen_dir, mat_file)
    if os.path.exists(mat_file_path):
        shutil.copy(mat_file_path, os.path.join(remainder_dir, mat_file))
