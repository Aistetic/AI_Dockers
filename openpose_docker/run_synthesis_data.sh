#!/bin/bash

export PYTHONPATH=:./

until docker-compose up --exit-code-from openpose_male; do
  echo "openpose crashed with exit code $?.  Respawning.." >&2
  sleep 1
done

