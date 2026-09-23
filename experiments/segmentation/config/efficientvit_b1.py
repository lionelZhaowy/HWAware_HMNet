"""Asynchronous B/C default; common implementation lives in async_config."""
from hmnet.utils.async_config import AsyncSettings

class TrainSettings(AsyncSettings):
    pass

class TestSettings(AsyncSettings):
    frame_evaluation = True
    batch_size = AsyncSettings.eval_batch_size
    def get_model(self, devices=None, mode="single_process"):
        if mode != "single_process":raise ValueError("Async evaluation requires single_process")
        return super().get_model()
