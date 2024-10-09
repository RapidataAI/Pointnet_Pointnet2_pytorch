import os

from ultralytics import YOLO
import wandb
from wandb.integration.ultralytics import add_wandb_callback

PROJECT = 'Points2BBox'
im_size = 128
user_score_threshold = 0.0
data_type = 'user_score_lines'
data_folder = (f'yolo_rapids_{data_type}_{im_size}x{im_size}'
               f'{f"_us{user_score_threshold}" if user_score_threshold is not None else ""}')
pretrained = True

model_name = 'yolov8n.pt' if pretrained else 'yolov8n.yaml'
if os.path.exists(model_name):
    os.remove(model_name)

wandb_runid = f'{data_type}_{im_size}_{"pretrained" if pretrained else "scratch"}'

wandb = wandb.init(project=PROJECT, job_type="training", name=wandb_runid)

model = YOLO(model_name)

add_wandb_callback(model, enable_model_checkpointing=True)

results = model.train(
    project=PROJECT,
    data=f"{data_folder}.yaml",
    epochs=300,
    imgsz=im_size,
    single_cls=True,
    pretrained=pretrained,
    name=wandb_runid,
    val=pretrained
)

wandb.finish()

