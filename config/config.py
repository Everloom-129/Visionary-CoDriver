import os
from dotenv import load_dotenv

load_dotenv()

API_TOKEN = os.getenv("DDS_CLOUDAPI_TEST_TOKEN")
MODEL = "GDino1_5_Pro"
DETECTION_TARGETS = ["Mask", "BBox"]