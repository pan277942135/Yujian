"""All unit tests run without outbound cloud/metadata/telemetry access."""
import os
from scripts.offline_test_guard import install, install_cloud_fakes

os.environ['ORT_DISABLE_TELEMETRY']='1'
install()
install_cloud_fakes()
