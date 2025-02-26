import kagglehub

# Download latest version
path = kagglehub.dataset_download("adikurniawan/color-dataset-for-color-recognition")

print("Path to dataset files:", path)