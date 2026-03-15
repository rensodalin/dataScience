import os
import numpy as np
import faiss
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# ------------------------
# Folders
# ------------------------
IMAGE_FOLDER = r"D:\DataScience\data\processed_images"
EMBED_FOLDER = r"D:\DataScience\embeddings"
os.makedirs(EMBED_FOLDER, exist_ok=True)

INDEX_PATH = os.path.join(EMBED_FOLDER, "product_index.faiss")
LABELS_PATH = os.path.join(EMBED_FOLDER, "labels.npy")
PATHS_PATH = os.path.join(EMBED_FOLDER, "paths.npy")

# ------------------------
# Load CLIP model
# ------------------------
print("🔄 Loading CLIP model...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

# ------------------------
# Build FAISS index if missing
# ------------------------
if not (os.path.exists(INDEX_PATH) and os.path.exists(LABELS_PATH) and os.path.exists(PATHS_PATH)):
    print("⚡ Building FAISS index...")

    embeddings = []
    labels = []
    paths = []

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

    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # Save index and metadata
    faiss.write_index(index, INDEX_PATH)
    np.save(LABELS_PATH, labels)
    np.save(PATHS_PATH, paths)

    print("✅ FAISS index created!")

else:
    print("⚡ Loading existing FAISS index...")
    index = faiss.read_index(INDEX_PATH)
    labels = np.load(LABELS_PATH, allow_pickle=True).tolist()
    paths = np.load(PATHS_PATH, allow_pickle=True).tolist()

# ------------------------
# Search function
# ------------------------
def get_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        output = model.get_image_features(**inputs)
        if isinstance(output, torch.Tensor):
            emb = output
        else:
            emb = getattr(output, "image_embeds", None) or getattr(output, "last_hidden_state", None)
        emb = emb.cpu().numpy().flatten()
    return emb

def search_similar(query_image_path, top_k=3):
    query_emb = get_image_embedding(query_image_path).astype("float32")
    query_emb = np.expand_dims(query_emb, axis=0)
    faiss.normalize_L2(query_emb)

    scores, indices = index.search(query_emb, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "product": labels[idx],
            "image": paths[idx],
            "similarity_score": float(score)
        })
    return results

# ------------------------
# Test search
# ------------------------
TEST_IMAGE = r"D:\DataScience\data\test_images\shampoo\test1.jpg"  # <-- Change to your test image
results = search_similar(TEST_IMAGE, top_k=3)

print("\n🔍 Top 3 similar products:")
for i, r in enumerate(results):
    print(f"{i+1}. Product: {r['product']} | Score: {r['similarity_score']:.3f} | Image: {r['image']}")