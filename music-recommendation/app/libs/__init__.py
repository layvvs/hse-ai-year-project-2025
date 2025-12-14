import sys
import os
import platform
arch = platform.machine()
sys.path.insert(1, os.path.join(os.path.dirname(__file__), arch))
