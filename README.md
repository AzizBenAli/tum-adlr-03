## Representing shapes as Latent Codes
> This project was conducted as part of the Advanced Deep Learning course at the Technical University of Munich. 
<img width="797" alt="Screenshot 2025-02-22 at 21 15 53" src="https://github.com/user-attachments/assets/53993287-57fd-451c-ae42-fcf2aef88c67" />

## 📌 Overview  

This project focuses on deep learning techniques for **robotics applications**, including:  
- **Latent space optimization**  
- **3D object representation**  
The implementation extends the **DeepSDF framework** and explores improvements in **multi-class training**.

## 📂 Repository Structure  
📂 project_root  
 ┣ 📂 configs/              
 ┃ ┣ 📜 hyperparameters.yaml  
 ┃ ┣ 📜 settings.yaml  
 ┣ 📂 multi_class/            
 ┃ ┣ 📂 data/               
 ┃ ┣ 📂 trained_models/     
 ┃ ┣ 📂 visualization/      
 ┣ 📂 scripts/             
 ┃ ┣ 📂 data_manipulation/  
 ┃ ┣ 📂 evaluation/         
 ┃ ┣ 📂 helpers/            
 ┃ ┣ 📂 models/             
 ┃ ┣ 📂 training/            
 ┃ ┣ 📂 utils/               
 ┣ 📜 README.md            
 ┣ 📜 requirements.txt     

## 🚀 Setup Instructions  

Follow these steps to set up and run the project:  

### 1️⃣ Create a Virtual Environment  
Run the following command to create and activate a virtual environment:  

```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```
### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Train and Evaluate the Model
```bash
python main.py --mode multi_class  # Multi-class training
python main.py --mode single_class # Single-class training
```

🖼️ Visual Results

<div style="display: flex; align-items: center;">
  <img src="[https://github.com/user-attachments/assets/a42e0828-a6d7-4afb-a5dd-44fcd0b64022](https://github.com/user-attachments/assets/a42e0828-a6d7-4afb-a5dd-44fcd0b64022)" alt="Image 1" style="max-width: 50%; height: auto; margin-right: 10px;">
  <video controls style="max-width: 50%; height: auto;">
    <source src="[https://github.com/user-attachments/assets/4b5c20b3-e607-49a9-8812-ead2f0425139](https://github.com/user-attachments/assets/4b5c20b3-e607-49a9-8812-ead2f0425139)" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>




