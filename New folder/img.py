import cv2
import numpy as np

def region_growing(img, seed_point, threshold):
    mask = np.zeros_like(img, dtype=np.uint8)  # Create an empty mask
    visited = np.zeros_like(img, dtype=np.uint8)  # Create a visited array to track visited pixels

    # Define connectivity (4 or 8 neighbors)
    connectivity = 4

    # Get seed point coordinates
    x, y = seed_point

    # Get seed point intensity
    seed_intensity = img[y, x]

    # Define a queue for pixel traversal
    queue = [(x, y)]

    # Set the visited flag for the seed point
    visited[y, x] = 1

    # Perform region growing
    while len(queue) > 0:
        # Get the next pixel coordinates from the queue
        x, y = queue.pop(0)

        # Set the pixel in the mask
        mask[y, x] = img[y, x]

        # Get the neighbors of the current pixel
        neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]

        # Process the neighbors
        for neighbor_x, neighbor_y in neighbors:
            # Check if the neighbor is within the image boundaries
            if 0 <= neighbor_x < img.shape[1] and 0 <= neighbor_y < img.shape[0]:
                # Check if the neighbor is unvisited and its intensity is within the threshold
                if np.any(visited[neighbor_y, neighbor_x]) == 0 and all(np.abs(img[neighbor_y, neighbor_x, channel] - seed_intensity[channel]) <= threshold[channel] for channel in range(img.shape[2])):



                    # Set the visited flag for the neighbor
                    visited[neighbor_y, neighbor_x] = 1

                    # Add the neighbor to the queue for further processing
                    queue.append((neighbor_x, neighbor_y))

    return mask

# Load the RGB image
img = cv2.imread("cameraman.jpg")

# Create a window and display the image
cv2.namedWindow("Image")
cv2.imshow("Image", img)

# Mouse callback function to get seed points
seed_points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        seed_points.append((x, y))
        cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
        cv2.imshow("Image", img)

# Register the mouse callback function
cv2.setMouseCallback("Image", mouse_callback)

# Wait for the user to click on the image
cv2.waitKey(0)
cv2.destroyAllWindows()

# Define the threshold
threshold = np.array([20, 20, 20])

# Apply region growing algorithm for each seed point
segmented_image = np.zeros_like(img, dtype=np.uint8)  # Create an empty segmented image
for seed_point in seed_points:
    mask = region_growing(img, seed_point, threshold)
    segmented_image = cv2.bitwise_or(segmented_image, mask)

# Save the segmented image
cv2.imwrite("segmented_image.jpg", segmented_image)

# Display the segmented image
cv2.imshow("Segmented Image", segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()