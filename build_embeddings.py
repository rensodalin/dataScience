
import os
import numpy as np
import faiss
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Folders
IMAGE_FOLDER = r"D:\DataScience\data\processed_images"
EMBED_FOLDER = r"D:\DataScience\embeddings"
os.makedirs(EMBED_FOLDER, exist_ok=True)

# Load CLIP model
print("🔄 Loading CLIP model...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

embeddings = []
labels = []
paths = []

# Loop over processed images
for product in os.listdir(IMAGE_FOLDER):
    product_folder = os.path.join(IMAGE_FOLDER, product)
    if not os.path.isdir(product_folder):
        continue

    for img_file in os.listdir(product_folder):
        if img_file.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(product_folder, img_file)
            print(f"Processing: {path}")
            image = Image.open(path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")

            with torch.no_grad():
                output = model.get_image_features(**inputs)
                # Extract the tensor
                if isinstance(output, torch.Tensor):
                    emb = output
                else:
                    emb = getattr(output, "image_embeds", None) or getattr(output, "last_hidden_state", None)
                if emb is None:
                    raise ValueError("Cannot find embedding tensor in CLIP output")

                emb = emb.cpu().numpy().flatten()

            embeddings.append(emb)
            labels.append(product)
            paths.append(path)

# Convert embeddings to FAISS
embeddings = np.array(embeddings).astype("float32")
faiss.normalize_L2(embeddings)

dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# Save index and metadata
faiss.write_index(index, os.path.join(EMBED_FOLDER, "product_index.faiss"))
np.save(os.path.join(EMBED_FOLDER, "labels.npy"), labels)
np.save(os.path.join(EMBED_FOLDER, "paths.npy"), paths)

print("✅ FAISS index created!")