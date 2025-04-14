import torch
from models.models import *
from torchvision import datasets, transforms
from torch.autograd import Variable

def round_two_sig_after_decimal(x):
    if x == 0:
        return 0.00
    int_part = int(abs(x))
    shift = 2 + len(str(int_part))
    return round(x, shift)

model = torch.load('model.pth', map_location=torch.device('cpu'), weights_only=False)

transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
])

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transform),
    batch_size=1000, shuffle=True)

model.eval()

correct = 0
for data, target in test_loader:
    data, target = Variable(data), Variable(target)
    original_target = target.cpu().detach().numpy()
    target = torch.where(target == 3, 
                        torch.tensor(1, device=target.device), 
                        torch.tensor(0, device=target.device))

    # Accumulate metrics
    output, fms, _, _ = model(data, 1, None)
    pred = output.data.max(1, keepdim=True)[1]
    correct += pred.eq(target.data.view_as(pred)).cpu().sum()

print(f'Accuracy: {round_two_sig_after_decimal(correct.item() / 100)}%')
