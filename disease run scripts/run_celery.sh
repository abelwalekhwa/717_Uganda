#!/bin/bash

# Run celery worker processes
STIDIR="$(dirname $(dirname $(realpath $0)))"
export PYTHONPATH="${PYTHONPATH}:${STIDIR}"
if [ "$HOSTNAME" = chinchilla120 ]; then
  export GAVI_OB_REDIS_URL="redis://chinchilla120:6379"
  CONCURRENCY=120
elif [ "$HOSTNAME" = agouti120 ]; then
  export GAVI_OB_REDIS_URL="redis://chinchilla120:6379"
  CONCURRENCY=120
elif [ "$HOSTNAME" = compute1 ]; then
  export GAVI_OB_REDIS_URL="redis://compute1:6379"
  CONCURRENCY=120
elif [ "$HOSTNAME" = compute2 ]; then
  export GAVI_OB_REDIS_URL="redis://compute1:6379"
  CONCURRENCY=120
elif [ "$HOSTNAME" = compute3 ]; then
  export GAVI_OB_REDIS_URL="redis://compute1:6379"
  CONCURRENCY=120
elif [ "$HOSTNAME" = athena ]; then
  export GAVI_OB_REDIS_URL="redis://apollo:6379"
  CONCURRENCY=30
elif [ "$HOSTNAME" = apollo ]; then
  export GAVI_OB_REDIS_URL="redis://apollo:6379"
  CONCURRENCY=30
elif [ "$HOSTNAME" = temp3 ]; then
  export GAVI_OB_REDIS_URL="redis://localhost:6378"
  CONCURRENCY=50
elif [ "$HOSTNAME" = IAZPVWKS010 ]; then
  export GAVI_OB_REDIS_URL="redis://localhost:6378"
  CONCURRENCY=50
elif [ "$HOSTNAME" = IAZPVWKS011 ]; then
  export GAVI_OB_REDIS_URL="redis://localhost:6378"
  CONCURRENCY=50
else
  export GAVI_OB_REDIS_URL="redis://localhost:6379"
  CONCURRENCY=4
fi
celery -A stisim.gavi.celery worker -l info -Q gavi-outbreaks -Ofair
