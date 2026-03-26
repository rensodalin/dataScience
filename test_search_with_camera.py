import os
import numpy as np
import faiss
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import cv2

EMBED_FOLDER = r"D:\CADT Year 3th\Term2\Data Science\dataScience\embeddings"
INDEX_PATH = os.path.join(EMBED_FOLDER, "product_index.faiss")
LABELS_PATH = os.path.join(EMBED_FOLDER, "labels.npy")
PATHS_PATH = os.path.join(EMBED_FOLDER, "paths.npy")

# Load CLIP model
print("🔄 Loading CLIP model...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

# Load FAISS index & metadata
index = faiss.read_index(INDEX_PATH)
labels = np.load(LABELS_PATH, allow_pickle=True).tolist()
paths = np.load(PATHS_PATH, allow_pickle=True).tolist()
print(f"✅ FAISS index loaded! Total images: {index.ntotal}")

# Functions
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

# Capture image from webcam
cap = cv2.VideoCapture(0)
print("📷 Press 'c' to capture an image, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Show live camera
    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1)

    if key == ord('c'):
        # Save webcam frame to file
        test_image_path = "test_capture.jpg"  # temporary file
        cv2.imwrite(test_image_path, frame)
        print(f"✅ Image captured: {test_image_path}")

        # Search using file path
        results = search_similar(test_image_path, top_k=3)
        print("\nTop 3 similar products:")
        for i, r in enumerate(results):
            print(f"{i+1}. Product: {r['product']} | Score: {r['similarity_score']:.3f} | Image: {r['image']}")

        # Show top images
        for r in results:
            img = Image.open(r['image'])
            img.show(title=r['product'])

        print("Done! Press 'c' again to capture another image, or 'q' to quit.")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()