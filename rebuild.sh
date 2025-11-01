#!/bin/bash

# This makes the script exit immediately if any command fails
set -e

# --- Configuration ---
IMAGE_NAME="credisight-api"
HOST_PORT=8080
CONTAINER_PORT=80
# ---------------------

echo "--- Looking for old '$IMAGE_NAME' containers ---"

# Find any container (running or stopped) that was built from our image
CONTAINER_ID=$(docker ps -a -q --filter ancestor=$IMAGE_NAME)

if [ -n "$CONTAINER_ID" ]; then
  echo "Stopping and removing existing container: $CONTAINER_ID"
  docker stop $CONTAINER_ID
  docker rm $CONTAINER_ID
else
  echo "No existing container found. Proceeding to build."
fi

echo "--- Building new image: $IMAGE_NAME ---"
docker build -t $IMAGE_NAME .

echo "--- Running new container on host port $HOST_PORT ---"
docker run -d -p $HOST_PORT:$CONTAINER_PORT $IMAGE_NAME

echo "---"
echo "Deployment complete! API should be live on port $HOST_PORT."
echo "You can check its logs with: docker logs $(docker ps -q)"