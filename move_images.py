import os
import shutil


main_folder = "/satassdscratch/amete7/xai-ood"  # Replace with the path to your main folder
imagenet_val_folder = os.path.join(main_folder, "imagenet_val")

# Create the "imagenet_val" folder if it doesn't exist
os.makedirs(imagenet_val_folder, exist_ok=True)

# Iterate through files in the main folder
for filename in os.listdir(main_folder):
    if filename.endswith(".JPEG") or filename.endswith(".jpg"):
        # Construct the source and destination paths
        src_path = os.path.join(main_folder, filename)
        dst_path = os.path.join(imagenet_val_folder, filename)

        # Move the file to the "imagenet_val" folder
        shutil.move(src_path, dst_path)

print("Files moved to 'imagenet_val' folder.")
