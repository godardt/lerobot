#!/bin/bash

vcam_ip="100.75.155.120"
vcam_local_path="/dev/video0"

echo -e "\e[32mVideo stream accessible at $vcam_local_path on the system\e[0m"
ffmpeg -i "http://$vcam_ip:4747/video" -vf "scale=640:480,fps=30,format=yuyv422" -f v4l2 "$vcam_local_path"
