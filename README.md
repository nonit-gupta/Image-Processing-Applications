# Steel Surface Defect Detection

This project focuses on the automatic detection and classification of surface defects on steel sheets using computer vision and deep learning techniques. The goal is to build a model that can accurately identify various types of defects from images.

## Project Structure

- `Steel_Defect_Detection_endeval.ipynb`: The main Jupyter Notebook containing the end-to-end process of data loading, model building, training, and evaluation.
- `assignment/assignment2_imageproc.ipynb`: A supplementary notebook, likely containing earlier explorations of image processing techniques.
- `Steel Surface Defect Detection_Report.docx`: A detailed report of the project.
- `requirements.txt`: A list of Python dependencies required to run the project.

## Dataset

The project uses a dataset of steel surface images, each labeled with a specific defect type. The dataset is organized into `train`, `validation`, and `test` directories. Annotations, including filenames and corresponding class labels, are provided in a `_annotations.csv` file within each directory.

The data loading pipeline is set up to read images and labels from these files and prepare them for training with TensorFlow.

## Methodology

Two distinct approaches were investigated to tackle the defect detection problem:

### 1. Baseline Convolutional Neural Network (CNN)

- **Description**: A standard CNN model was built to serve as a baseline for performance.
- **Architecture**:
  - Input Layer: (128, 128, 3)
  - Conv2D Layer (32 filters, 3x3 kernel, ReLU)
  - MaxPooling2D Layer (2x2)
  - Conv2D Layer (64 filters, 3x3 kernel, ReLU)
  - MaxPooling2D Layer (2x2)
  - Flatten Layer
  - Dense Layer (128 units, ReLU)
  - Dropout Layer (0.5)
  - Output Dense Layer (softmax activation for multiple classes)
- **Training**: The model was trained for 10 epochs using the Adam optimizer. To address the problem of class imbalance in the dataset, class weights were calculated and applied during training.
- **Results**: This model achieved a test accuracy of **72.5%**.

### 2. CNN with Defect Segmentation

- **Description**: This approach attempted to improve performance by first segmenting the defect from the background before feeding the image to the classifier.
- **Segmentation Preprocessing**:
  - Images were first converted to grayscale.
  - Canny edge detection was applied to find the edges of the potential defect.
  - Morphological operations (closing and dilation) were used to create a clean binary mask of the segmented defect.
  - This mask was then applied to the original image, effectively isolating the defect.
- **Architecture**: The same CNN architecture as the baseline model was used.
- **Training**: The model was trained on the pre-processed, segmented images.
- **Results**: This approach yielded a test accuracy of **68.8%**, which was lower than the baseline model.

## Results and Conclusion

| Model                               | Test Accuracy |
| ----------------------------------- | :-----------: |
| Baseline CNN                        |   **72.5%**   |
| CNN with Defect Segmentation        |     68.8%     |

The results indicate that the baseline CNN model performed better than the model trained on segmented images. The segmentation process, while intended to help the model focus on the defect, may have inadvertently removed useful contextual information from the images, leading to a slight degradation in performance.

The final model provides a solid baseline for steel defect detection. Further improvements could be explored by using more complex architectures (like ResNet, VGG), extensive data augmentation, or more advanced segmentation techniques.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Install dependencies:**
    Make sure you have Python installed. Then, install the required packages using pip.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up the dataset:**
    The notebook `Steel_Defect_Detection_endeval.ipynb` expects the dataset to be located in a specific Google Drive folder path (`/content/drive/MyDrive/Dataset/`). You will need to:
    - Upload the dataset to your Google Drive.
    - Or, modify the paths in the notebook (Cell 4) to point to the location of your `train`, `valid`, and `test` data directories.

4.  **Run the notebook:**
    Launch Jupyter Notebook and open `Steel_Defect_Detection_endeval.ipynb`.
    ```bash
    jupyter notebook
    ```
    You can then run the cells sequentially to see the entire workflow, from data loading to model evaluation. 
