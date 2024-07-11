from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

# Load the image
image_path = "path"  # Update this with the path to your image
image = Image.open(image_path).convert("RGB")

# Initialize the processor and model
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

# Define candidate labels
candidate_labels = ["LABELS"]

# Process the image and run the model
inputs = processor(text=candidate_labels, images=image, return_tensors="pt")
outputs = model(**inputs)

# Extract boxes and scores
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process(outputs=outputs, target_sizes=target_sizes)

# Display the image with detections
fig, ax = plt.subplots(1, figsize=(12, 12))
ax.imshow(image)

# Add bounding boxes to the image
for i, (box, label_index, score) in enumerate(zip(results[0]["boxes"], results[0]["labels"], results[0]["scores"])):
    label = candidate_labels[label_index]
    score = score.item()

    if score > 0.005:  # Display only high confidence detections
        # Convert box to numpy array after detaching
        box = box.detach().numpy()
        # Create a rectangle patch
        rect = patches.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=2,
            edgecolor='r',
            facecolor='none'
        )

        # Add the patch to the Axes
        ax.add_patch(rect)
        ax.text(box[0], box[1] - 10, f"{label} ({score:.2f})", color='red', fontsize=12, weight='bold')

plt.axis('off')  # Hide axes
plt.show()

