This repo stores the code for, "Identifying EEG Biomarkers of Depression with Novel Explainable Deep Learning Architectures".

The data used in the study is publicly available at:
https://figshare.com/articles/dataset/EEG_Data_New/4244171

An anaconda environment can be set up with:
requirements.txt.

The data can be preprocessed using:
Preprocess_Data.py.

Models M1 through M3 can be trained with:
Train_M1_KerasTuner.py
Train_M2_KerasTuner.py
Train_M3_KerasTuner.py

Statistical analyses comparing M1 through M3 model performance and explainability analyses can be performed with:
Comparative_Analysis_All_Models.ipynb

Novel explainability analyses of M2 and M3 can be performed with:
Comparative_Analysis_M2_M3.ipynb
