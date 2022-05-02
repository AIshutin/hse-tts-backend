# Acoustic service
text -> mels
    cd ./acoustic_service/.

## To launch acoustic service local:
    flask run  
or  
    python3 app.py

## To launch acoustic service in Docker:
### Build container
    docker build . -t fastpitch:latest  
### Start container
    docker run --gpus=all --name acoustic_service -e CUDA_VISIBLE_DEVICES -it --ipc=host -p 5000:5000 fastpitch:latest
### Stop container
    docker stop acoustic_service
### remove container 
    docker rm acoustic_service 

## Example of requests:
    python3 check.py