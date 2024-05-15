import torch

SEED = 2024
torch.manual_seed(2024)

model = torch.jit.load("model.pth")

sample_examples = torch.rand((1, 3, 518, 686)).to(torch.float32)

with torch.no_grad():
    output = model(sample_examples)
    print(output)
