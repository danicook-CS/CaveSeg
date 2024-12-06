import cv2
import numpy as np
from PIL import Image
import os
def mask_to_yolo_format(mask_path):
    """
    Convert a segmentation mask to segmentation polygons in YOLO format.

    :param mask_path: Path to the segmentation mask image
    :param class_id: Class ID for the objects in the mask
    :return: List of YOLO formatted polygons
    """
    mask = np.array(Image.open(mask_path).convert('L'))

    annotations = []

    # Find unique object ids in the mask
    unique_labels = np.unique(mask)

    # Remove background (assuming background label is 0)
    unique_labels = unique_labels[unique_labels != 0]
    for label in unique_labels:
        # Create a binary mask for the current label
        if label == 22:
            class_id = 1
        elif label == 34:
            class_id = 2
        elif label == 38:
            class_id = 3
        elif label == 52:
            class_id = 4
        elif label == 53:
            class_id = 5
        elif label == 57:
            class_id = 6
        elif label == 64:
            class_id = 7
        elif label == 75:
            class_id = 8
        elif label == 76:
            class_id = 9
        elif label == 90:
            class_id = 10
        elif label == 170:
            class_id = 11
        elif label == 189:
            class_id = 12
        elif label == 192:
            class_id = 13
        mask_label = np.uint8(mask == label)
        # Find contours (polygons) of the binary mask
        contours, _ = cv2.findContours(mask_label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if contour.size >= 6:  # Ensure there are enough points to form a polygon
                contour = contour.flatten().tolist()

                # Normalize the polygon coordinates
                normalized_contour = []
                for i in range(0, len(contour), 2):
                    x = contour[i] / mask.shape[1]
                    y = contour[i + 1] / mask.shape[0]
                    normalized_contour.extend([x, y])

                # YOLO format: class_id, polygon_points...
                annotation = [class_id-1] + normalized_contour
                annotations.append(annotation)

    return annotations

def save_yolo_format(annotations, output_path):
    """
    Save annotations in YOLO format to a file.

    :param annotations: List of annotations in YOLO format
    :param output_path: Path to the output file
    """
    with open(output_path, 'w') as f:
        for annotation in annotations:
            annotation_str = ' '.join(map(str, annotation))
            f.write(f"{annotation_str}\n")

# Example usage
image_path = "/home/afrl/newImages/05757.jpg"
output_path = "/home/afrl/mmsegmentation/output_test"
mask_path = '/data/mreza/CaveSegmentation/data/labels2/CaveSegChallenge'
for item in os.listdir(image_path):
    name = item[:-4]
    label_name = name + '.png'
    name = name + '.txt'
    label = os.path.join(mask_path, label_name)
    output_txt = os.path.join(output_path, name)
    annotations = mask_to_yolo_format(label)
    save_yolo_format(annotations, output_txt)
