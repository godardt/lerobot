#!/bin/bash

echo -e "\e[32mStarting teleoperation...\e[0m"

# Often required
sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

cd "$DIR/.."

# Start Rerun in headless mode in the background
echo -e "\e[32mStarting Rerun in headless mode...\e[0m"
#pixi run rerun -- --bind 0.0.0.0:9876 --no-viewer &
#RERUN_PID=$!

# Wait a moment for Rerun to start
sleep 2

# Runs headless
export DISPLAY=
#pixi run python lerobot/scripts/control_robot.py --robot.type=so100 --control.type=teleoperate
pixi run python lerobot/scripts/control_robot.py \
	--robot.type=so100 \
	--control.type=teleoperate \
	--control.display_data=true \
	--remote_robot.viewer_ip=100.110.81.39 \
	--remote_robot.viewer_port=9876

# Supposedly use the following on the client `rerun connect rerun+http://100.101.174.46:9876`

# Cleanup: kill Rerun when the script exits
kill $RERUN_PID
