import importlib
import logging


def get_safe_default_codec():
    if importlib.util.find_spec("torchcodec"):
        return "torchcodec"
    else:
        logging.warning("'torchcodec' is not available in your platform, falling back to 'pyav' as a default decoder")
        return "pyav"


def make_cameras_from_configs(cameras_config):
    cameras = {}
    for cam_name, cam_config in cameras_config.items():
        fps = cam_config["fps"]
        width = cam_config["width"]
        height = cam_config["height"]
        rotation = cam_config["rotation"]

        if cam_config["type"] == "ip_camera":
            ip = cam_config["vcam_ip"]
            port = cam_config["vcam_port"]
            cameras[cam_name] = IPCamera(ip, port, fps, width, height, rotation)
        elif cam_config["type"] == "opencv":
            device = cam_config["device"]
            cameras[cam_name] = OpenCVCamera(device, fps, width, height, rotation)

    return cameras
