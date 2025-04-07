from dds_cloudapi_sdk import Config, Client, DetectionTask, TextPrompt, DetectionModel, DetectionTarget
from dds_cloudapi_sdk.tasks.ivp import TaskResult
from .config import API_TOKEN, MODEL, DETECTION_TARGETS

class APIClient:
    def __init__(self):
        config = Config(API_TOKEN)
        self.client = Client(config)

    def detect_objects(self, image_path, prompts):
        image_url = self.client.upload_file(image_path)
        task = DetectionTask(
            image_url=image_url,
            prompts=[TextPrompt(text=prompt) for prompt in prompts],
            targets=[getattr(DetectionTarget, target) for target in DETECTION_TARGETS],
            model=getattr(DetectionModel, MODEL),
        )
        self.client.run_task(task)
        return task.result

    def rle2rgba(self, rle_mask):
        # Create a dummy task with minimal required arguments
        dummy_task = DetectionTask(
            image_url="dummy",
            prompts=[TextPrompt(text="dummy")],
            targets=[DetectionTarget.Mask],
            model=getattr(DetectionModel, MODEL)
        )
        return dummy_task.rle2rgba(rle_mask)