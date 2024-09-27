#!/bin/bash
#cd mmpose
cd /mmpose/projects/rtmpose3d/
wget -O /mmpose/projects/rtmpose3d/entrypoint.py https://raw.githubusercontent.com/lanmower/3d-tracker/main/docker/entrypoint.py
python entrypoint.py
