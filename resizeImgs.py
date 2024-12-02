import os
import cv2

# Directory containing the original images
input_directory = './Symbols/'
# Directory to save the resized images
output_directory = './ResizedSymbols64/'

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Loop through all the files in the input directory
for filename in os.listdir(input_directory):
    # Only process image files (you can customize the condition)
    if filename.endswith(".png") or filename.endswith(".jpg"):
        # Construct full file path
        input_path = os.path.join(input_directory, filename)
        output_path = os.path.join(output_directory, filename)

        # Read the image using OpenCV
        img = cv2.imread(input_path)

        # Resize the image to 20x20
        img_resized = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)

        # Save the resized image to the output directory
        cv2.imwrite(output_path, img_resized)

        print(f"Resized image saved: {output_path}")
