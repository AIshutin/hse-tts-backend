# Telegram bot service
cd ./telegram_bot_service/.
## Quick start  

    cd ./telegram_bot_service/.
## To launch Telegram bot service local:
    python3 main.py

## To launch Telegram bot service in Docker:
### Build container
    docker build . -t telegram_bot_service:latest
### Start container
    docker run --name telegram_bot_service -e CUDA_VISIBLE_DEVICES -it --ipc=host -p 5001:5001 telegram_bot_service:latest
### Stop container
    docker stop telegram_bot_service
### remove container 
    docker rm telegram_bot_service 

## Bot commands:  
For all users:
Start chat:   

    /start  
For developers only:
Get analitics:  

    /analitics    
    
Testing:
    /latency_test
