import logging
from pathlib import Path

from tuatini.common.motors import FeetechMotorsBus
from tuatini.common.video import start_smartphone_stream


class SO100Robot:
    def __init__(self, config):
        self.config = config
        # Start the video stream in a separate process
        try:
            vcam_ip, vcam_port, vcam_output_path = self._get_smartphone_config()
            logging.info(f"Connecting to smartphone camera from {vcam_ip}:{vcam_port} to {vcam_output_path}")
            stream_process = start_smartphone_stream(vcam_ip, vcam_port, vcam_output_path)
        except KeyboardInterrupt:
            logging.info("Shutting down smartphone camera stream...")
        except Exception:
            logging.error("Failed to start smartphone stream, ignoring...")
        finally:
            # Clean up the process
            if stream_process and stream_process.is_alive():
                stream_process.terminate()
                stream_process.join()

        # TODO check connection to the wrist cam

        leader_arms = self.make_motors_buses_from_configs(self.config["leader_arms"], type="feetch")
        follower_arms = self.make_motors_buses_from_configs(self.config["follower_arms"], type="feetch")
        cameras = self.make_cameras_from_configs(self.config["cameras"])

        self.calibration_dir = Path(self.config["calibration_dir"])

    def _get_smartphone_config(self):
        """Get viewer IP and port from config file."""
        smartphone_config = self.config["cameras"]["front"]
        ip = smartphone_config["vcam_ip"]
        port = smartphone_config["vcam_port"]
        output_path = smartphone_config["camera_index"]
        return ip, port, output_path

    def make_motors_buses_from_configs(self, motors_bus_configs, type):
        motors_buses = {}

        for key, data in motors_bus_configs.items():
            if type == "feetch":
                motors = data["motors"]
                port = data["port"]
                motors_buses[key] = FeetechMotorsBus(motors, port)
            else:
                raise ValueError(f"The motor type '{type}' is not valid.")

        return motors_buses

    def make_cameras_from_configs(self, config):
        pass

    def teleop_step(self, record_data=False):
        pass
