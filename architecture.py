import torch
from generator.encoder_model import SwinTransformer3D

model = SwinTransformer3D()
# input = (batch_size, channels, frames, height, width)
dummy_x = torch.rand(2, 3, 4, 224, 224)
logits, skip_connections = model(dummy_x)
# print(logits.shape)
for i, item in enumerate(skip_connections):
  print(item.shape, f'skip connection {i+1}')

# print(model)