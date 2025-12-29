# End-to-end-Youtube-Sentiment
![MLOps](https://img.shields.io/badge/Project-MLOps-blueviolet?style=for-the-badge&logo=git)
![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![AWS](https://img.shields.io/badge/AWS-Cloud-orange?style=for-the-badge&logo=amazon-aws)
![DVC](https://img.shields.io/badge/DVC-Data_Version_Control-white?style=for-the-badge&logo=dvc)

## 👥 Team Members
* **Manel BOUMADANE** (Project Lead & Data Engineer)
* **Oulahcene Anissa** (Machine Learning Engineer)
* **Ider Tilleli** (DevOps & Frontend Engineer)

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















conda create -n youtube python=3.11 -y 

conda activate youtube

pip install -r requirements.txt


## DVC

dvc init

dvc repro

dvc dag



## AWS

aws configure



### Json data demo in postman

http://localhost:5000/predict

```python
{
    "comments": ["This video is awsome! I loved a lot", "Very bad explanation. poor video"]
}
```



chrome://extensions


## how to get youtube api key from gcp:

https://www.youtube.com/watch?v=i_FdiQMwKiw



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

    ECR_REPOSITORY_NAME = simple-app
