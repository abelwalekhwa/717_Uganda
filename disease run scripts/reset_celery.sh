#!/bin/bash

# Delete any lingering jobs in the Celery queue
STIDIR="$(dirname $(dirname $(dirname $(realpath $0))))"
export PYTHONPATH="${PYTHONPATH}:${STIDIR}"
if [ "$HOSTNAME" = chinchilla120 ]; then
  export GAVI_OB_REDIS_URL="redis://chinchilla120:6379"
elif [ "$HOSTNAME" = agouti120 ]; then
  export GAVI_OB_REDIS_URL="redis://chinchilla120:6379"
elif [ "$HOSTNAME" = athena ]; then
  export GAVI_OB_REDIS_URL="redis://apollo:6379"
elif [ "$HOSTNAME" = apollo ]; then
  export GAVI_OB_REDIS_URL="redis://apollo:6379"
elif [ "$HOSTNAME" = temp3 ]; then
  export GAVI_OB_REDIS_URL="redis://localhost:6378"
elif [ "$HOSTNAME" = IAZPVWKS010 ]; then
  export GAVI_OB_REDIS_URL="redis://localhost:6378"
elif [ "$HOSTNAME" = IAZPVWKS011 ]; then
  export GAVI_OB_REDIS_URL="redis://localhost:6378"
else
  export GAVI_OB_REDIS_URL="redis://localhost:6379"
fi
celery -A stisim.gavi.celery purge
