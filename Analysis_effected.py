import cv2
import numpy as np
import matplotlib.pyplot as plt


def analyze_paddy_leaf(image_path):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found")
        return

    # Convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to HSV color
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define color ranges
    lower_healthy = np.array([30, 40, 40])
    upper_healthy = np.array([90, 255, 255])

    # Create mask
    healthy_mask = cv2.inRange(hsv, lower_healthy, upper_healthy)

    # Invert
    affected_mask = cv2.bitwise_not(healthy_mask)
    kernel = np.ones((5, 5), np.uint8)
    affected_mask = cv2.morphologyEx(affected_mask, cv2.MORPH_OPEN, kernel)
    affected_mask = cv2.morphologyEx(affected_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(affected_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours
    result = img_rgb.copy()
    cv2.drawContours(result, contours, -1, (255, 0, 0), 2)

    # Calculate affected percentage
    total_pixels = img.shape[0] * img.shape[1]
    affected_pixels = cv2.countNonZero(affected_mask)
    affected_percent = (affected_pixels / total_pixels) * 100


    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(healthy_mask, cmap='gray')
    plt.title('Healthy Areas Mask')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(affected_mask, cmap='gray')
    plt.title('Affected Areas Mask')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(result)
    plt.title(f'Detected Affected Areas ({affected_percent:.2f}%)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Additional analysis
    print(f"Percentage of affected area: {affected_percent:.2f}%")
    print(f"Number of affected spots: {len(contours)}")

    # Severity classification
    if affected_percent < 10:
        print("Severity: Mild")
    elif affected_percent < 30:
        print("Severity: Moderate")
    else:
        print("Severity: Severe")

image_path = 'PestEffected/ef.jpg'
analyze_paddy_leaf(image_path)