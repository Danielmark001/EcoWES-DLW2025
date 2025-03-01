import os

import numpy as np
import paramiko


def connect(server="5.35.252.104", username="rutherford"):
    """ Connect to server via ssh
    """
    ssh = paramiko.SSHClient()
    ssh.load_host_keys(
        os.path.expanduser(os.path.join("~", ".ssh", "known_hosts"))
    )
    ssh.connect(server, username)
    sftp = ssh.open_sftp()

    return sftp, ssh


# vectorized haversine function
def haversine(lat1, lon1, lat2, lon2, to_radians=True, earth_radius=6371):
    """
    slightly modified version: of http://stackoverflow.com/a/29546836/2901002

    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees or in radians)

    All (lat, lon) coordinates must have numeric dtypes and be of equal length.

    """

    if to_radians:
        lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])

    a = (
        np.sin((lat2 - lat1) / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2
    )

    return earth_radius * 2 * np.arcsin(np.sqrt(a))
