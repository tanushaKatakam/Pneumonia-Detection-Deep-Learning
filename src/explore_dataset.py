import os
import matplotlib.pyplot as plt

from config import TRAIN_DIR

normal_path = os.path.join(TRAIN_DIR, "NORMAL")
pneumonia_path = os.path.join(TRAIN_DIR, "PNEUMONIA")

normal_count = len(os.listdir(normal_path))
pneumonia_count = len(os.listdir(pneumonia_path))

print("Training Dataset Statistics")
print("---------------------------")
print(f"NORMAL     : {normal_count}")
print(f"PNEUMONIA : {pneumonia_count}")

plt.figure(figsize=(6,5))

plt.bar(
    ["NORMAL", "PNEUMONIA"],
    [normal_count, pneumonia_count]
)

plt.title("Training Dataset Distribution")
plt.ylabel("Number of Images")

plt.show()