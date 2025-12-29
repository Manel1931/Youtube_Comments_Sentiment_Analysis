<!-- # End-to-end-Youtube-Sentiment
![MLOps](https://img.shields.io/badge/Project-MLOps-blueviolet?style=for-the-badge&logo=git)
![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![AWS](https://img.shields.io/badge/AWS-Cloud-orange?style=for-the-badge&logo=amazon-aws)
![DVC](https://img.shields.io/badge/DVC-Data_Version_Control-white?style=for-the-badge&logo=dvc)

## 👥 Team Members
* **Manel BOUMADANE**
* **Oulahcene Anissa**
* **Ider Tilleli**

---

## 🎯 Project Purpose
This **YouTube Sentiment Analysis** project is a comprehensive **MLOps pipeline** designed to scrape, process, and analyze the sentiment of YouTube comments in real-time. By leveraging a custom-trained LightGBM model and a Chrome extension, users can instantly visualize audience feedback on any video. This project demonstrates the integration of data versioning with DVC, experiment tracking with MLflow, and automated cloud deployment using AWS.

---

## 🛠️ Tech Stack & Tools
- **Language:** Python 3.11
- **ML Frameworks:** LightGBM, Scikit-learn
- **Data Versioning:** DVC (Data Version Control)
- **Experiment Tracking:** MLflow
- **Cloud Infrastructure:** AWS (S3, ECR, EC2)
- **DevOps:** GitHub Actions (CI/CD), Docker
- **Frontend:** Chrome Extension (JavaScript/HTML/CSS)







## 📊 MLflow on AWS Setup
This project uses AWS to host a dedicated experiment tracking server.

### 1. Infrastructure Preparation
* **Login to AWS console.**
* **Create IAM user** with `AdministratorAccess`.
* **Export credentials** in your local terminal by running `aws configure`.
* **Create an S3 bucket** to store model artifacts.
* **Create EC2 machine** (Ubuntu) and update Security Groups to allow **Port 5000**.

### 2. EC2 Server Configuration
Run these commands on your EC2 instance to initialize the tracking server:
```bash
sudo apt update && sudo apt install python3-pip virtualenv -y
mkdir mlflow && cd mlflow
# Setup environment
pip install mlflow awscli boto3
aws configure

# Start the MLflow Server
mlflow server \
    -h 0.0.0.0 \
    --default-artifact-root s3://your-s3-bucket-name \
    --allowed-hosts *


After starting **set your tracking URI in your local terminal:** 'export MLFLOW_TRACKING_URI=http://your-ec2-public-ip:5000'




## 🚀 Getting Started Locally

## 1. Create a New Conda Environnement with Python 3.11

conda create -n youtube python=3.11 -y 

conda activate youtube

pip install -r requirements.txt


## 2. Data Version Control (DVC)

dvc init

dvc repro # Run the full pipeline

dvc dag # Visualize pipeline dependency graph


## 3. Usage & API
The Flask server provides a prediction endpoint 
Postman Demo:
```python
{
    "comments": ["This video is awsome! I loved a lot", "Very bad explanation. poor video"]
}
```

## AWS

aws configure



### Json data demo in postman

http://localhost:5000/predict





chrome://extensions


# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


	#Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

	
## 3. Create ECR repo to store/save docker image
    - Save the URI: 315865595366.dkr.ecr.us-east-1.amazonaws.com/youtube

	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
# 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one


# 7. Setup github secrets:

    AWS_ACCESS_KEY_ID=

    AWS_SECRET_ACCESS_KEY=

    AWS_REGION = eu-west-3

    AWS_ECR_LOGIN_URI = demo>>  566373416292.dkr.ecr.ap-south-1.amazonaws.com

    ECR_REPOSITORY_NAME = simple-app -->




# 📊 End-to-End YouTube Sentiment Analysis (MLOps Project)

![MLOps](https://img.shields.io/badge/Project-MLOps-blueviolet?style=for-the-badge&logo=git)
![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![AWS](https://img.shields.io/badge/AWS-Cloud-orange?style=for-the-badge&logo=amazon-aws)
![DVC](https://img.shields.io/badge/DVC-Data_Version_Control-white?style=for-the-badge&logo=dvc)

## 👥 Team Members
* **Manel BOUMADANE**
* **Oulahcene Anissa**
* **Ider Tilleli**

---

## 🎯 Project Purpose
This **YouTube Sentiment Analysis** project is a comprehensive **MLOps pipeline** designed to scrape, process, and analyze the sentiment of YouTube comments in real-time. By leveraging a custom-trained LightGBM model and a Chrome extension, users can instantly visualize audience feedback on any video. This project demonstrates the integration of data versioning with DVC, experiment tracking with MLflow, and automated cloud deployment using AWS.

---

## 🛠️ Tech Stack & Tools
- **Language:** Python 3.11
- **ML Frameworks:** LightGBM, Scikit-learn
- **Data Versioning:** DVC (Data Version Control)
- **Experiment Tracking:** MLflow
- **Cloud Infrastructure:** AWS (S3, ECR, EC2)
- **DevOps:** GitHub Actions (CI/CD), Docker
- **Frontend:** Chrome Extension (JavaScript/HTML/CSS)

---

## 🚀 Getting Started Locally

## 1. Create a New Conda Environment with Python 3.11
```bash
conda create -n youtube python=3.11 -y 
conda activate youtube
pip install -r requirements.txt
```

## 2. Data Version Control (DVC)
```bash
dvc init
dvc repro # Run the full pipeline
dvc dag   # Visualize pipeline dependency graph
```

## 3. Usage & API
The Flask server provides a prediction endpoint at http://localhost:5000/
* Postman Demo JSON Body: 
```bash
{
    "comments": ["This video is awsome! I loved a lot", "Very bad explanation. poor video"]
}
```


## 📊 MLflow on AWS Setup
This project uses AWS to host a dedicated experiment tracking server.

# 1. Infrastructure Preparation
```bash
Login to AWS console.

Create IAM user with AdministratorAccess.

Export credentials in your local terminal by running "aws configure".

Create an S3 bucket to store model artifacts.

Create EC2 machine (Ubuntu) and update Security Groups to allow Port 5000.
```

# 2. EC2 Server Configuration
Run these commands on your EC2 instance to initialize the tracking server:
```bash
sudo apt update && sudo apt install python3-pip virtualenv -y
mkdir mlflow && cd mlflow
# Setup environment
pip install mlflow awscli boto3
aws configure

# Finally, Start the MLflow Server
mlflow server \
    -h 0.0.0.0 \
    --default-artifact-root s3://your-s3-bucket-name \
    --allowed-hosts *
```

* After starting, set your tracking URI in your local terminal: 
```bash
export MLFLOW_TRACKING_URI=http://your-ec2-public-ip:5000
```











## ☁️ AWS-CICD-Deployment-with-Github-Actions
* Login to AWS console.
* Create IAM user for deployment


#with specific access

**1. EC2 access :** It is virtual machine

**2. ECR:** Elastic Container registry to save your docker image in aws

# Description: About the deployment

1. Build docker image of the source code

2. Push your docker image to ECR

3. Launch Your EC2

4. Pull Your image from ECR in EC2

5. Launch your docker image in EC2

**#Policy:**

**AmazonEC2ContainerRegistryFullAccess**

**AmazonEC2FullAccess**

## 3. Create ECR repo to store/save docker image
Save the URI

## 4. Create EC2 machine (Ubuntu)

## 5. Open EC2 and Install docker in EC2 Machine:
**#optional**
```bash
sudo apt-get update -y
sudo apt-get upgrade
```

**#required**
```bash
curl -fsSL [https://get.docker.com](https://get.docker.com) -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
```


## 6. Configure EC2 as self-hosted runner:
**setting > actions > runner > new self hosted runner > choose os > then run command one by one**

## 7. Setup github secrets:
**AWS_ACCESS_KEY_ID = ...**

**AWS_SECRET_ACCESS_KEY = ...**

**AWS_REGION = eu-west-3**

**AWS_ECR_LOGIN_URI = ...**

**ECR_REPOSITORY_NAME = ...**

## 🌐 Chrome Extension Setup
1. Navigate to **chrome://extensions**

2. Enable Developer **Mode**.

3. Click **Load unpacked** and select the extension folder.