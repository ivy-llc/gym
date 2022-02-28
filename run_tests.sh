#!/bin/bash -e
docker run --rm -it -v "$(pwd)":/ivy_gym unifyai/ivy-gym:latest python3 -m pytest ivy_gym_tests/
