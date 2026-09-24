"""One configuration implementation for both independent asynchronous projects."""
from hmnet.models.efficientvit_tasks import ROOT,build_frame_task
from hmnet.dataset.dsec_async import DSECAsync
from hmnet.utils.config import load_config

variant = load_config(str(ROOT/"experiments/segmentation/config/async_variant.py"),"async_variant")


class AsyncSettings:
    frame_training = True
    task = "segmentation"
    modality = "rgbdvs"
    fusion_mode = "add"
    temporal_window = 2
    precision = "bf16"
    epochs = 150
    updates = None
    batch_size = 32
    accumulation = 1
    workers = 8
    prefetch_factor = 1
    eval_batch_size = 32
    learning_rate = 2e-4
    min_learning_rate = 2e-6
    lr_schedule = "warmup_cosine"
    warmup_epochs = 5
    warmup_start_factor = .1
    eval_every_epochs = 1
    weight_decay = .01
    pretrained = str(ROOT/"pretrained/efficientvit_b1_r224.pth")
    window_us,bins = (50000,10) if variant.VARIANT == "B" else (25000,5)
    cache = "/home/zhaowenyao24/Conda_prj/lab_dataset/DSEC_Semantic/preprocessed/async_v1/dsec_async_"+variant.VARIANT
    output = str(ROOT/"logs/segmentation"/f"efficientvit_b1_{variant.VERSION}_stage1")
    resume = ""
    init_from = None
    pseudo_root = None
    allow_failed_pseudo_audit = False
    pseudo_weight = .2
    pseudo_ramp_epochs = 5
    overfit = 0
    deterministic = False
    validation_limit = None
    def get_model(self):
        return build_frame_task("segmentation",pretrained=self.pretrained,modality="rgbdvs",
                                fusion_mode="add",temporal_window=2,event_channels=self.bins*2)
    def get_dataset(self):
        return DSECAsync(self.cache,"train",clips=True,augment=True,window_us=self.window_us,bins=self.bins,
            pseudo_root=self.pseudo_root,pseudo_weight=self.pseudo_weight,pseudo_ramp_epochs=self.pseudo_ramp_epochs,diagnostic_pseudo=getattr(self,"diagnostic_pseudo",False),
            allow_failed_pseudo_audit=self.allow_failed_pseudo_audit)
    def get_validation_dataset(self):
        return DSECAsync(self.cache,"dev",window_us=self.window_us,bins=self.bins,limit=self.validation_limit)
