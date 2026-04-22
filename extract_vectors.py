import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import struct

# MobileNet V2 — fast, good enough, no GPU needed
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

BATCH_SIZE = 64
NUM_SAMPLES = 10000  # full CIFAR-10 train set
OUTPUT_FILE = "vectors.bin"
LABELS_FILE = "labels.bin"

transform = transforms.Compose([
    transforms.Resize(224),           # MobileNet expects 224x224
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet stats, standard for pretrained models
        std=[0.229, 0.224, 0.225]
    )
])

print("Loading CIFAR-10...")
dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)

loader = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

print("Loading MobileNet V2...")
model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

# Remove the classifier head — keep only the feature extractor
# Output of features is (batch, 1280, 7, 7) → after pooling → (batch, 1280)
# We'll use adaptive avg pool to get a flat vector, then take first 512 dims
feature_extractor = torch.nn.Sequential(
    model.features,
    torch.nn.AdaptiveAvgPool2d((1, 1)),
    torch.nn.Flatten()
)
feature_extractor.eval()

all_vectors = []
all_labels = []

print("Extracting features...")
with torch.no_grad():
    for i, (images, labels) in enumerate(loader):
        features = feature_extractor(images)   # shape: (batch, 1280)
        features = features[:, :512]            # take first 512 dims
        all_vectors.append(features.numpy())
        all_labels.append(labels.numpy())
        if (i + 1) % 20 == 0:
            print(f"  Processed {(i+1)*BATCH_SIZE}/{NUM_SAMPLES} images")

all_vectors = np.concatenate(all_vectors, axis=0).astype(np.float32)
all_labels  = np.concatenate(all_labels,  axis=0).astype(np.int32)

print(f"Vector matrix shape: {all_vectors.shape}")   # should be (10000, 512)

# Write binary file: header then raw floats
# Format: [int32: num_vectors][int32: dims][float32 x num_vectors x dims]
with open(OUTPUT_FILE, "wb") as f:
    f.write(struct.pack("ii", all_vectors.shape[0], all_vectors.shape[1]))
    f.write(all_vectors.tobytes())

# Save labels separately (for validation later)
with open(LABELS_FILE, "wb") as f:
    f.write(struct.pack("i", all_labels.shape[0]))
    f.write(all_labels.tobytes())

print(f"Saved {all_vectors.shape[0]} vectors to {OUTPUT_FILE}")
print(f"Saved {all_labels.shape[0]} labels  to {LABELS_FILE}")
print(f"File size: {all_vectors.nbytes / 1e6:.1f} MB")