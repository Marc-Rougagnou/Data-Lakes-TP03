#!/bin/bash

# Check if a LocalStack container is already running
if [ "$(docker ps -q -f name=amazing_galileo)" ]; then
    echo "LocalStack is already running."
else
    # Check if the container exists but is stopped
    if [ "$(docker ps -aq -f name=amazing_galileo)" ]; then
        echo "LocalStack container exists but is stopped. Restarting..."
        docker start amazing_galileo
    else
        echo "LocalStack container does not exist. Starting a new container..."
        docker run -d -p 4566:4566 -p 4572:4572 --name amazing_galileo localstack/localstack
    fi
fi
