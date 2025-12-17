import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/root/arob_ws/src/arob_lab_drones/install/arob_lab_drones'
