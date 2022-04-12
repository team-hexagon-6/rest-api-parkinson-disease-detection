# Parkinson Disease Detection Rest API


## Installation 

>**Save Repository:** `https://github.com/team-hexagon-6/rest-api-parkinson-disease-detection.git`
####
>**Install Packages:** `pip install joblib,opencv-python,scikit-image,numpy,flask,flask_restfull,flask_cors`


## Run Server

>**Development Environment:** `run app.py`
####
>**Production Environment:** `flask run`


## API access

>**Localhost:** `http://127.0.0.1`
####
>**PORT:** `5000`
####
>**API URL:** `http://127.0.0.1:5000/api`
####
>**Authentication:** `user_id and access_token as query_parameters`
####
>**Data:** `image and type`

#
##  Authentication Requirement Example
>**user_id:**`1234`
####
>**access_token:**`f7a50a548fbccdfc1d960d9f3f97ecfe241205b9a35d9fbefd562828f5e34265`
####
>**API URL query parameters:** `http://127.0.0.1:5000/api?user_id=1234&access_token=f7a50a548fbccdfc1d960d9f3f97ecfe241205b9a35d9fbefd562828f5e34265`


## Data Sending Example
>**Data:** `image data should be send as form data base64 encoded`
