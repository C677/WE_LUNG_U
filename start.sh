#!/bin/sh
# change path to start.sh
cd $(cd "$(dirname "$0")" && pwd)

# make file
echo "path=""$(pwd)" > .env

# run bash shell of weLungU container
docker-compose run --rm -p 5000:5000 weLungU bash