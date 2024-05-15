# import torch

# # Load the dictionary containing the model parameters
# checkpoint = torch.load("FastSAM-s.pt", map_location=torch.device('cpu'))  # Load the checkpoint

# # If the model was saved with DataParallel, remove the 'module.' prefix
# if 'module.' in list(checkpoint['model'].keys())[0]:
#     state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
# else:
#     state_dict = checkpoint['model']

# # Instantiate your custom model class
# model = CustomCNN()  # Instantiate your actual custom model class (CustomCNN)

# # Load the model state dictionary
# model.load_state_dict(state_dict)

# # Set the model to evaluation mode
# model.eval()

# # Define a function to infer input size from the model architecture
# def infer_input_size(model):
#     input_shape = next(iter(model.parameters())).shape
#     input_size = (1,) + tuple(input_shape[1:])  # Assuming the first dimension is batch size
#     return input_size

# # Infer input size
# input_size = infer_input_size(model)

# # Create a dummy input tensor
# dummy_input = torch.randn(input_size)

# # Export the PyTorch model to ONNX
# torch.onnx.export(model, dummy_input, "Fastsam.onnx", verbose=True)

# print("Model converted successfully to ONNX format.")

# from model import FastSAM
import torch
import onnx
import torch.onnx

# torch_model = FastSAM()
# torch_input = torch.randn(1, 1,1920,1080)
# onnx_program = torch.onnx.dynamo_export(torch_model, torch_input)

# onnx_program.save("my_image_classifier.onnx")

# onnx_model = onnx.load("my_image_classifier.onnx")
# onnx.checker.check_model(onnx_model)
model_dict=torch.load(r'/mnt/c/Users/saket/OneDrive_UCSD/Documents/Research/FastSAM/weights/FastSAM-x.pt')
model=model_dict['model']

model=model.float()

input=torch.randn(1,3,768,1024)
print(input.shape)
output='/mnt/c/Users/saket/OneDrive_UCSD/Documents/Research/FastSAM/weights/FastSAM-x.onnx'
torch.onnx.export(model,input,output,verbose=True)