import os
import cv2
import numpy as np
import re
import random
import matplotlib.pyplot as plt
import csv

class DataPreprocessor:
    def __init__(self, data_folder):
        """
        Initialize the DataPreprocessor class.

        Parameters:
        - data_folder (str): The path to the folder containing the images.
        """
        self.data_folder = data_folder
        self.images = []  # Will store tuples of (filename, mask)
        self.sdf_maps = {}  # Will store SDF maps for each image
        self.points_info = {}  # Will store selected points and their SDF values

    def load_images(self):
        """
        Loads images that represent angle '20' from the specified folder, maintaining the order.
        """
        # Define a flexible regex pattern to match filenames containing '-20.' or '-20-'
        angle_pattern = re.compile(r'.*[_\-]20[_\-].*|.*[_\-]20\..*')

        # List all files in the data folder
        all_files = os.listdir(self.data_folder)
        print(f"Total files found in {self.data_folder}: {len(all_files)}")

        # Filter files matching the angle_pattern and valid extensions
        matched_files = []
        for f in all_files:
            if angle_pattern.match(f) and f.lower().endswith(('.png', '.jpg', '.jpeg')):
                matched_files.append(f)
            else:
                # Uncomment the next line to see which files are not matched
                # print(f"Not matched: {f}")
                pass

        print(f"Files matching the pattern: {len(matched_files)}")
        for f in matched_files:
            print(f" - {f}")

        # Sort the matched files
        filenames = sorted(matched_files)

        # Load the images
        for filename in filenames:
            image_path = os.path.join(self.data_folder, filename)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is not None:
                self.images.append((filename, image))
            else:
                print(f"Failed to load image: {filename}")
        print(f"Loaded {len(self.images)} images at angle '20'.")

    def binarize_images(self):
        """
        Traces the object's outer border, highlights it, fills within the borders with white,
        and sets the background to black.
        """
        processed_images = []
        for filename, image in self.images:
            # Convert to grayscale
            grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian Blur to reduce noise
            blurred = cv2.GaussianBlur(grayscale_image, (5, 5), 0)

            # Perform edge detection using Canny with specified thresholds
            edges = cv2.Canny(blurred, threshold1=10, threshold2=20)

            # Dilate the edges to close gaps
            kernel = np.ones((3, 3), np.uint8)
            dilated_edges = cv2.dilate(edges, kernel, iterations=1)

            # Find contours from the edges
            contours, hierarchy = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Create an empty mask
            mask = np.zeros_like(grayscale_image)

            # If contours were found
            if contours:
                # Find the largest contour by area (assuming it's the object)
                largest_contour = max(contours, key=cv2.contourArea)

                # Draw the contour on the mask (highlight the border)
                cv2.drawContours(mask, [largest_contour], -1, color=255, thickness=2)

                # Fill the interior of the contour
                cv2.drawContours(mask, [largest_contour], -1, color=255, thickness=cv2.FILLED)
            else:
                print(f"No contours found in image: {filename}")

            # The mask now has the object filled with white and the background is black
            processed_images.append((filename, mask))

        self.images = processed_images
        print("Traced object's outer border, highlighted it, and filled within the borders with white.")

    def compute_sdf(self):
        """
        Computes the Signed Distance Function (SDF) for each binary mask.
        """
        for filename, mask in self.images:
            # Normalize the mask to have values 0 and 1
            binary_norm = mask / 255.0

            # Compute the distance transform for the background and the object
            dist_transform_inside = cv2.distanceTransform((binary_norm).astype(np.uint8), cv2.DIST_L2, 5)
            dist_transform_outside = cv2.distanceTransform((1 - binary_norm).astype(np.uint8), cv2.DIST_L2, 5)

            # Compute the signed distance function
            sdf = dist_transform_outside - dist_transform_inside

            # Store the SDF map
            self.sdf_maps[filename] = sdf

        print("Computed Signed Distance Function (SDF) for each binary mask.")

    def select_points(self, num_total_points=100, border_percentage=0.7):
        """
        Selects points near the object's border as well as some from inside and outside.

        Parameters:
        - num_total_points (int): Total number of points to select per image.
        - border_percentage (float): Percentage of points to select near the border.
        """
        points_info = {}  # Stores selected points and their SDF values for each image

        for filename, sdf in self.sdf_maps.items():
            # Calculate number of points for each category
            num_border = int(num_total_points * border_percentage)
            num_inside = int((num_total_points - num_border) / 2)
            num_outside = num_total_points - num_border - num_inside

            # Points near the border (|SDF| <= threshold_border)
            threshold_border = 5
            border_indices = np.argwhere(np.abs(sdf) <= threshold_border)

            # Points inside the object (SDF < -threshold_inside)
            threshold_inside = threshold_border
            inside_indices = np.argwhere(sdf <= -threshold_inside)

            # Points outside the object (SDF > threshold_outside)
            threshold_outside = threshold_border
            outside_indices = np.argwhere(sdf >= threshold_outside)

            # Randomly select points from each category
            selected_border = self._random_select(border_indices, num_border)
            selected_inside = self._random_select(inside_indices, num_inside)
            selected_outside = self._random_select(outside_indices, num_outside)

            # Combine all selected points
            if selected_border.size == 0 and selected_inside.size == 0 and selected_outside.size == 0:
                print(f"No points available for selection in image: {filename}")
                continue

            selected_points = np.vstack((selected_border, selected_inside, selected_outside)) if selected_border.size else \
                              np.vstack((selected_inside, selected_outside)) if selected_inside.size else \
                              selected_outside

            selected_sdf_values = sdf[selected_points[:, 0], selected_points[:, 1]]
            point_labels = ['inside' if sdf_val < 0 else 'outside' for sdf_val in selected_sdf_values]

            # Store the points and their labels
            points_info[filename] = {
                'points': selected_points,
                'sdf_values': selected_sdf_values,
                'labels': point_labels
            }

        print(f"Selected up to {num_total_points} points for each image (70% near border, 15% inside, 15% outside).")
        self.points_info = points_info

    def _random_select(self, indices, num_select):
        """
        Helper function to randomly select a specified number of points from given indices.

        Parameters:
        - indices (numpy.ndarray): Array of point indices.
        - num_select (int): Number of points to select.

        Returns:
        - selected (numpy.ndarray): Selected point indices.
        """
        if indices.size == 0:
            return np.empty((0, 2), dtype=int)
        num_select = min(num_select, indices.shape[0])
        selected = indices[np.random.choice(indices.shape[0], num_select, replace=False)]
        return selected

    def save_points_to_csv(self, output_folder):
        """
        Saves the selected points and their SDF values to CSV files.

        Each CSV file is named after the corresponding image.

        Parameters:
        - output_folder (str): The path where CSV files should be saved.
        """
        csv_folder = os.path.join(output_folder, 'csv_files')
        os.makedirs(csv_folder, exist_ok=True)

        for filename, info in self.points_info.items():
            points = info['points']
            sdf_values = info['sdf_values']
            labels = info['labels']

            csv_filename = os.path.splitext(filename)[0] + '.csv'
            csv_path = os.path.join(csv_folder, csv_filename)

            # Write to CSV file with column names
            with open(csv_path, mode='w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                # Write header
                csv_writer.writerow(['x', 'y', 'sdf', 'location'])
                # Write data rows
                for (row, col), sdf_value, label in zip(points, sdf_values, labels):
                    csv_writer.writerow([col, row, sdf_value, label])  # Note: x is col, y is row

        print(f"Saved points and SDF values to CSV files in {csv_folder}.")

    def annotate_and_save_images(self, output_folder):
        """
        Annotates all selected points on the processed images.

        Points inside and outside are distinguished by color.

        Parameters:
        - output_folder (str): The path where annotated images should be saved.
        """
        annotated_folder = os.path.join(output_folder, 'annotated_images')
        os.makedirs(annotated_folder, exist_ok=True)

        for filename, mask in self.images:
            # Convert binary mask to BGR for annotation
            annotated_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            # If there are selected points for this image
            if filename in self.points_info:
                points = self.points_info[filename]['points']
                labels = self.points_info[filename]['labels']

                for (row, col), label in zip(points, labels):
                    if label == 'inside':
                        # Green color for inside points
                        cv2.circle(annotated_image, (col, row), radius=3, color=(0, 255, 0), thickness=-1)
                    else:
                        # Blue color for outside points
                        cv2.circle(annotated_image, (col, row), radius=3, color=(255, 0, 0), thickness=-1)

            # Save the annotated image
            output_path = os.path.join(annotated_folder, f"annotated_{filename}")
            cv2.imwrite(output_path, annotated_image)

        print(f"Annotated images saved in {annotated_folder}.")

    def plot_random_annotated_images(self, output_folder, num_images=20):
        """
        Plots 20 random pairs of original images and their annotated counterparts.

        - Left Cluster: 4x5 grid of original images.
        - Right Cluster: 4x5 grid of annotated images.

        Parameters:
        - output_folder (str): The path where annotated images are saved.
        - num_images (int): Number of image pairs to plot (default is 20).
        """
        import matplotlib
        # Ensure there are enough images to plot
        total_available = len(self.images)
        if total_available == 0:
            print("No images available to plot.")
            return

        num_images = min(num_images, total_available)
        selected_images = random.sample(self.images, num_images)

        # Calculate grid dimensions
        rows = 4
        cols = 5
        max_images = rows * cols
        if num_images > max_images:
            print(
                f"Number of images to plot ({num_images}) exceeds the grid capacity ({max_images}). Adjusting to {max_images}.")
            num_images = max_images
            selected_images = selected_images[:max_images]

        # Create a figure with two clusters: originals and annotated
        fig, axes = plt.subplots(nrows=rows, ncols=cols * 2, figsize=(40, 30))  # Adjust figsize as needed
        plt.subplots_adjust(wspace=0.02, hspace=0.2)  # Reduce horizontal and vertical space

        for idx, (filename, mask) in enumerate(selected_images):
            # Load original image from data_folder
            original_image_path = os.path.join(self.data_folder, filename)
            original_image = cv2.imread(original_image_path, cv2.IMREAD_COLOR)
            if original_image is None:
                print(f"Failed to load original image: {original_image_path}")
                continue
            original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

            # Path to the annotated image
            annotated_image_path = os.path.join(output_folder, 'annotated_images', f"annotated_{filename}")
            if not os.path.exists(annotated_image_path):
                print(f"Annotated image not found: {annotated_image_path}")
                continue

            # Load annotated image
            annotated_image = cv2.imread(annotated_image_path, cv2.IMREAD_COLOR)
            if annotated_image is None:
                print(f"Failed to load annotated image: {annotated_image_path}")
                continue
            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

            # Determine subplot positions
            row = idx // cols
            col = idx % cols

            # Plot original image on the left cluster
            ax_orig = axes[row, col * 2]
            ax_orig.imshow(original_image_rgb)
            ax_orig.axis('off')  # Remove axis

            # Plot annotated image on the right cluster
            ax_annot = axes[row, col * 2 + 1]
            ax_annot.imshow(annotated_image_rgb)
            ax_annot.axis('off')  # Remove axis

        # Maximize the figure window before showing
        try:
            manager = plt.get_current_fig_manager()
            # For TkAgg backend (common on Windows)
            if isinstance(manager, matplotlib.backend_managers.TkAggManager):
                manager.window.state('zoomed')  # Maximize window on Windows
            # For Qt backend (common on Linux and some macOS setups)
            elif isinstance(manager, matplotlib.backend_managers.Qt4AggManager) or isinstance(manager,
                                                                                              matplotlib.backend_managers.Qt5AggManager):
                manager.window.showMaximized()
            # For GTK backend
            elif isinstance(manager, matplotlib.backend_managers.Gtk3AggManager):
                manager.window.maximize()
            # For WX backend
            elif isinstance(manager, matplotlib.backend_managers.WxAggManager):
                manager.window.Maximize()
            # For MacOS (using Cocoa backend)
            elif 'macosx' in matplotlib.get_backend():
                manager.window.set_fullscreen(True)
        except Exception as e:
            print(f"Could not maximize the window: {e}")
            # As a fallback, use full_screen_toggle
            try:
                fig.canvas.manager.full_screen_toggle()
            except AttributeError:
                print("Full screen toggle not supported for this backend.")

        plt.tight_layout()
        plt.show()

    def save_preprocessed_images(self, output_folder):
        """
        Save binary masks to the specified output folder, maintaining the original order.

        Parameters:
        - output_folder (str): The path where binary masks should be saved.
        """
        masks_folder = os.path.join(output_folder, 'binary_masks')
        os.makedirs(masks_folder, exist_ok=True)
        for filename, mask in self.images:
            output_path = os.path.join(masks_folder, f"processed_{filename}")
            cv2.imwrite(output_path, mask)
        print(f"Saved all preprocessed binary masks to {masks_folder}.")

    def preprocess(self, output_folder):
        """
        Executes the full preprocessing pipeline.

        Parameters:
        - output_folder (str): The path where binary masks, annotated images, and CSV files should be saved.
        """
        self.load_images()
        self.binarize_images()
        self.compute_sdf()
        self.select_points()
        self.save_preprocessed_images(output_folder)
        self.annotate_and_save_images(output_folder)
        self.save_points_to_csv(output_folder)
        self.plot_random_annotated_images(output_folder, num_images=20)  # Added this line

    def get_points_info(self):
        """
        Returns the selected points and their SDF values.

        Returns:
        - points_info (dict): Dictionary containing points and their SDF values for each image.
        """
        return self.points_info

    def visualize_points(self, filename, sdf, points):
        """
        Visualizes the selected points on the SDF map.

        Parameters:
        - filename (str): Name of the image file.
        - sdf (numpy.ndarray): The SDF map.
        - points (numpy.ndarray): Array of point coordinates.
        """
        plt.figure(figsize=(6, 6))
        plt.imshow(sdf, cmap='jet')
        plt.colorbar()
        plt.scatter(points[:, 1], points[:, 0], c='white', s=10)
        plt.title(f"Selected Points Near Border in {filename}")
        plt.show()

# Usage
if __name__ == "__main__":
    data_folder = "/Users/yahyaabdelhamed/Documents/tum-adlr-03/data/coil-100"  # Update this path to your data folder
    output_folder = "output"  # Folder to save processed images, annotated images, and CSV files

    # Verify that data_folder exists
    if not os.path.exists(data_folder):
        print(f"Error: The data folder does not exist: {data_folder}")
        exit(1)

    preprocessor = DataPreprocessor(data_folder)
    preprocessor.preprocess(output_folder)

    # Get the selected points and their SDF values
    points_info = preprocessor.get_points_info()

    # Example: Print points and labels for the first image
    for filename in points_info:
        print(f"\nImage: {filename}")
        print("Selected Points (x, y):")
        print(points_info[filename]['points'])
        print("Labels (inside/outside):")
        print(points_info[filename]['labels'])
        break  # Remove this line to print for all images