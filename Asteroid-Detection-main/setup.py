import os

file_path = r"C:\Users\Rakesh\OneDrive\Documents\Asteroid\asteroid\Asteroid-Detection-main\asteroid_process.txt"
if os.path.exists(file_path):
    print("File found!")
else:
    print("File not found.")
