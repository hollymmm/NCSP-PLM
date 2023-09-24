# NCSP-PLM
NCSP-PLM was developed for the prediction of non-classical secreted proteins.    
![image]([https://github.com/ZhiliangMa/MPU6500-HMC5983-AK8975-BMP280-MS5611-10DOF-IMU-PCB/blob/main/img/IMU-V5-TOP.jpg](https://github.com/hollymmm/NCSP-PLM/blob/master/1.jpg))    


# Requirements

numpy==1.16.2  
pandas==1.1.5  
scikit_learn==1.3.1  
tensorflow_gpu==1.15.5  
# Usage
Run "independent_vote.py" to generate ensemble results for the model on independent data.

Run "5_folds_vote.py" to obtain ensemble results for the model's five-fold cross-validation.

Run "independent_feature_deep.py" to get results for three different models using nine different embeddings on an independent dataset.

Run "5_folds_feature_deep.py" to obtain results for three different models using nine different embeddings in a five-fold cross-validation setup.



