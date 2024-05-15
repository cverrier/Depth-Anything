import IPython
import numpy as np
import requests
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

SEED = 2024
torch.manual_seed(SEED)

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf")

# prepare image for the model
# inputs = image_processor(images=image, return_tensors="pt")

# image_size = (640, 480)  # Magic numbers

sample_examples = torch.rand((1, 3, 518, 686)).to(torch.float32)

with torch.no_grad():
    print(model(sample_examples).predicted_depth)
    traced = torch.jit.trace(model, (sample_examples), strict=False)
    # outputs = model(sample_examples)
    # predicted_depth = outputs.predicted_depth


torch.jit.save(traced, "model.pth")

# # interpolate to original size
# prediction = torch.nn.functional.interpolate(
#     predicted_depth.unsqueeze(1),
#     size=image_size[::-1],
#     mode="bicubic",
#     align_corners=False,
# )

# # visualize the prediction
# output = prediction.squeeze().cpu().numpy()
# formatted = (output * 255 / np.max(output)).astype("uint8")
# depth = Image.fromarray(formatted)
# depth.show()
