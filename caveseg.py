import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from tkinter import Tk, filedialog

def mask_to_yolo_format(mask_path):
    """Convert a segmentation mask to segmentation polygons in YOLO format."""
    mask = np.array(Image.open(mask_path).convert('L'))
    annotations = []

    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels != 0]  # Remove background (0)

    for label in unique_labels:
        class_id = {
            22: 1, 34: 2, 38: 3, 52: 4, 53: 5,
            57: 6, 64: 7, 75: 8, 76: 9, 90: 10,
            170: 11, 189: 12, 192: 13
        }.get(label, -1)

        if class_id == -1:
            continue  # Skip unknown labels

        mask_label = np.uint8(mask == label)
        contours, _ = cv2.findContours(mask_label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if contour.size >= 6:
                contour = contour.flatten().tolist()
                normalized_contour = []

                for i in range(0, len(contour), 2):
                    x = contour[i] / mask.shape[1]
                    y = contour[i + 1] / mask.shape[0]
                    normalized_contour.extend([x, y])

                annotation = [class_id - 1] + normalized_contour
                annotations.append(annotation)

    return annotations

def save_yolo_format(annotations, output_path):
    """Save annotations in YOLO format to a file."""
    with open(output_path, 'w') as f:
        for annotation in annotations:
            annotation_str = ' '.join(map(str, annotation))
            f.write(f"{annotation_str}\n")

def select_images():
    """Open a file dialog to select images."""
    Tk().withdraw()  # Hides the root Tkinter window
    file_paths = filedialog.askopenfilenames(
        title="Select images",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    return list(file_paths)

def apply_segmentation(image, masks, class_ids):
    """Applies segmentation masks to the image."""
    for mask, class_id in zip(masks, class_ids):
        color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
                 (255, 0, 255), (0, 255, 255), (128, 0, 128), (128, 128, 0),
                 (0, 128, 128), (0, 0, 128), (255, 128, 0), (128, 255, 0), 
                 (0, 128, 255)][class_id % 13]

        # Resize the mask to match the original image size
        mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Create a colored mask
        colored_mask = np.zeros_like(image)
        colored_mask[mask_resized > 0] = color

        # Blend the mask with the image
        image = cv2.addWeighted(image, 1, colored_mask, 0.5, 0)

    return image

def run_yolo_on_images(model_path, image_paths, output_dir="results"):
    """Runs YOLO model on selected images and saves the results."""
    model = YOLO(model_path)
    os.makedirs(output_dir, exist_ok=True)  # Create output directory
    class_names = model.names

    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load {image_path}. Skipping...")
            continue

        results = model(image_path)[0]  # Get the first batch of results
        masks = results.masks.data.cpu().numpy() if results.masks is not None else []
        class_ids = results.boxes.cls.cpu().numpy().astype(int) if results.boxes is not None else []

        # Apply segmentation masks to the image
        segmented_image = apply_segmentation(image, masks, class_ids)

        output_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, segmented_image)
        print(f"Processed {image_path}, results saved to {output_path}")

def main():
    # Select the YOLO model path
    model_path = "best.pt"  # Change this to your model path

    # Select images to process
    image_paths = select_images()
    if not image_paths:
        print("No images selected. Exiting...")
        return

    # Process masks and run YOLO on the selected images
    for image_path in image_paths:
        mask_path = os.path.splitext(image_path)[0] + '.png'  # Assuming mask is same name as image
        if os.path.exists(mask_path):
            annotations = mask_to_yolo_format(mask_path)
            save_yolo_format(annotations, os.path.splitext(image_path)[0] + '.txt')
            print(f"Processed mask for {image_path}")
        else:
            print(f"Mask file does not exist for {image_path}. Skipping mask processing.")

    run_yolo_on_images(model_path, image_paths)

if __name__ == "__main__":
    main()

