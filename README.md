# Classification-of-Cervical-cancer-using-Resnet-50

This project focuses on automated classification of cervical cells from pap-smear images using deep learning, specifically a ResNet-50 convolutional neural network. The system aims to assist pathologists in early detection of cervical cancer by classifying cervical cells into five categories based on their morphology.

#Published Work:  
([https://ieeexplore.ieee.org/document/10425807]([url](https://ieeexplore.ieee.org/document/10425807)))

##Project Overview
Manual analysis of pap-smear images is labor-intensive and prone to human error. Automating this process with deep learning enables faster, more accurate screening, which is critical for early detection of cervical cancer.

This project uses the **SIPAKMED dataset**, which contains labeled pap-smear images, and applies **ResNet-50** for feature extraction. The model classifies cells into five types:

* **Superficial/Intermediate** – Normal epithelial cells
*  **Parabasal** – Immature basal cells
*  **Koilocytic**– Cells showing HPV-related changes
*  **Metaplastic** – Cells undergoing transformation
*  **Dyskeratotic** – Abnormal, pre-cancerous cells
The model analyzes key visual patterns such as cell structure, shape, and morphology, achieving high accuracy in classification.

##Features

1. **Deep Learning Architecture:** Utilizes pretrained ResNet-50 for robust feature extraction.
2. **High Accuracy:** Achieved 97.5% accuracy on test data using softmax-based classification.
3. **Data Preprocessing:** Includes resizing, normalization, and augmentation to improve model generalization.
4. **Visualization:** Highlights features contributing to classification for interpretability.
5. **Reproducible Research:** Fully implemented in Python with commonly used libraries.

##Tech Stack

**Language:** Python 3.x
**Libraries:** NumPy, OpenCV, TensorFlow / Keras, Matplotlib, Scikit-learn
**Model Architecture:** ResNet-50 (pretrained with fine-tuning)
**Dataset:** SIPAKMED pap-smear images
**Development Tools:** Jupyter Notebook, VS Code

##Dataset

The SIPAKMED dataset contains labeled cervical cell images across five categories. Each image is preprocessed before being fed into the model:
* Resized to 224x224 pixels
* Normalized pixel values
* Data augmentation (rotation, flipping) applied to improve generalization

##Model Architecture

**Backbone:** ResNet-50 pretrained on ImageNet
**Classification Head:** Fully connected layer followed by softmax for 5-class classification
**Optimizer:** Adam
**Loss Function:** Categorical crossentropy
**Metrics:** Accuracy
The model leverages ResNet-50’s deep residual connections to learn hierarchical visual features and achieves robust performance on small biomedical datasets.

##Results

**Test Accuracy:** 97.5%
Confusion matrix indicates strong performance across all five cell types.
Model successfully identifies morphological patterns important for cervical cancer screening.

**Sample Confusion Matrix:**
Superficial/Intermediate: 98%
Parabasal: 96%
Koilocytic: 97%
Metaplastic: 95%
Dyskeratotic: 98%
Visualization: The model can highlight regions of interest in cells to improve interpretability.

##Future Improvements

* Integrate attention mechanisms for more precise focus on cell nuclei and cytoplasm
* Expand dataset to include more patient samples and diverse imaging conditions
* Develop a web-based interface or mobile app for real-time screening
* Implement ensemble models or transformer-based architectures for further accuracy improvement
* Add automated report generation for pathologists
