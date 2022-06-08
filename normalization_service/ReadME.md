# Normalization service
raw text -> normalized text

  cd ./normalization_service/.
  ## To launch normalization service local:
    flask run  
or  

    python3 run_predict.py
    
## To launch normalization service in Docker:
### Build container
    docker build . --tag=norm # from /home/aishutin/normalization_service  
### Start container
    sudo docker run --gpus all -p 9997:9997 norm
### Stop container
    docker stop normalization_service
### remove container 
    docker rm normalization_service 

## Example of requests:
    python3 test.py
