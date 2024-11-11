from utils import *

class DataPreprocessor:
    def __init__(self, data_folder, output_folder, plot=False):
        self.data_folder = data_folder
        self.output_folder = output_folder
        self.plot = plot
        self.images = []
        self.sdf_maps = {}
        self.points_info = {}


    def process_images(self):
        self.images = load_images(self.data_folder)
        self.images = binarize_images(self.images)
        self.sdf_maps = compute_sdf(self.images)

    def generate_save_points_to_csv(self, num_total_points=100, border_percentage=0.7):
        self.points_info = select_points(self.sdf_maps, num_total_points, border_percentage)
        save_points_to_csv(self.points_info, self.output_folder)

    def save_preprocessed_images(self):
        masks_folder = os.path.join(self.output_folder, 'binary_masks')
        os.makedirs(masks_folder, exist_ok=True)
        for filename, mask in self.images:
            output_path = os.path.join(masks_folder, f"processed_{filename}")
            cv2.imwrite(output_path, mask)
        print(f"Saved all preprocessed binary masks to {masks_folder}.")

    def save_annotated_images(self):
        annotate_and_save_images(self.images, self.points_info, self.output_folder)

    def show_annotated_images(self, num_images=20):
        plot_random_annotated_images(self.images, self.data_folder, self.output_folder, num_images=num_images)

    def preprocess(self):
        self.process_images()
        self.generate_save_points_to_csv()
        self.save_preprocessed_images()
        self.save_annotated_images()
        if self.plot:
         self.show_annotated_images(num_images=20)

if __name__ == "__main__":
    data_folder = "../data/coil-100"
    output_folder = "../processed_data_folder"

    if not os.path.exists(data_folder):
        print(f"Error: The data folder does not exist: {data_folder}")
        exit(1)

    preprocessor = DataPreprocessor(data_folder, output_folder, plot=False)
    preprocessor.preprocess()

