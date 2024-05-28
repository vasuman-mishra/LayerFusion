import torch
import torch.nn as nn
import time

# Define the original model
class OriginalModel(nn.Module):
    def __init__(self):
        super(OriginalModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x

# Define the fused model
class FusedModel(nn.Module):
    def __init__(self, original_model):
        super(FusedModel, self).__init__()
        self.conv1 = self.fuse_conv_bn(original_model.conv1, original_model.bn1)
        self.relu1 = nn.ReLU()
        self.conv2 = self.fuse_conv_bn(original_model.conv2, original_model.bn2)
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        return x
    
    def fuse_conv_bn(self, conv, bn):
        with torch.no_grad():
            # Initialize a new convolutional layer with the same parameters
            fused_conv = nn.Conv2d(conv.in_channels,
                                   conv.out_channels,
                                   kernel_size=conv.kernel_size,
                                   stride=conv.stride,
                                   padding=conv.padding,
                                   bias=True)
            
            # Extract the batch normalization parameters
            gamma = bn.weight
            beta = bn.bias
            mean = bn.running_mean
            var = bn.running_var
            eps = bn.eps

            # Extract the convolutional parameters
            W = conv.weight
            if conv.bias is None:
                b = torch.zeros_like(mean)
            else:
                b = conv.bias

            # Reshape the batch normalization parameters for broadcasting
            gamma = gamma.view(-1, 1, 1, 1)
            beta = beta.view(-1)
            mean = mean.view(-1, 1, 1, 1)
            var = var.view(-1, 1, 1, 1)

            # Fuse the weights and biases
            W_fused = W * (gamma / torch.sqrt(var + eps))
            b_fused = beta + (b - mean.squeeze()) * (gamma.squeeze() / torch.sqrt(var.squeeze() + eps))

            # Copy the fused parameters to the new convolutional layer
            fused_conv.weight.copy_(W_fused)
            fused_conv.bias.copy_(b_fused)
        
        return fused_conv

# Instantiate models
original_model = OriginalModel()
fused_model = FusedModel(original_model)

# Move models to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
original_model.to(device)
fused_model.to(device)

# Generate random input
input_tensor = torch.randn(1, 3, 64, 64).to(device)

# Function to measure inference time
def measure_inference_time(model, input_tensor, iterations=1000):
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        for _ in range(iterations):
            _ = model(input_tensor)
        end_time = time.time()
    return (end_time - start_time) / iterations

# Measure inference time for original model
original_time = measure_inference_time(original_model, input_tensor)
print(f"Original model inference time: {original_time:.6f} seconds")

# Measure inference time for fused model
fused_time = measure_inference_time(fused_model, input_tensor)
print(f"Fused model inference time: {fused_time:.6f} seconds")
a=((original_time-fused_time)/original_time)*100
print(f"The decrease in the inference time: {a}%")

# Verify outputs are the same
original_output = original_model(input_tensor)
fused_output = fused_model(input_tensor)
print(torch.allclose(original_output, fused_output, atol=1e-6))  # Should be True

