from ultralytics import YOLO
import wandb
from wandb.integration.ultralytics import add_wandb_callback

training_type = 'yolo_rapids_line_encode'
wandb.init(project=training_type, job_type="training")
model = YOLO("yolov8n.pt")


add_wandb_callback(model, enable_model_checkpointing=True)
results = model.train(project=training_type, data=f"{training_type}.yaml", epochs=500, imgsz=64)

wandb.finish()
