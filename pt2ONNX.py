import torch.onnx 

# #Function to Convert to ONNX 
# def Convert_ONNX(): 

#     # set the model to inference mode 
#     model.eval() 

#     # Let's create a dummy input tensor  
#     dummy_input = torch.randn(1, 512, requires_grad=True)  

#     # Export the model   
#     torch.onnx.export(model,         # model being run 
#          dummy_input,       # model input (or a tuple for multiple inputs) 
#          "ImageClassifier.onnx",       # where to save the model  
#          export_params=True,  # store the trained parameter weights inside the model file 
#          opset_version=10,    # the ONNX version to export the model to 
#          do_constant_folding=True,  # whether to execute constant folding for optimization 
#          input_names = ['modelInput'],   # the model's input names 
#          output_names = ['modelOutput'], # the model's output names 
#          dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
#                        'modelOutput' : {0 : 'batch_size'}}) 
#     print(" ") 
#     print('Model has been converted to ONNX') 

#Function to Convert to ONNX 
def Convert_ONNX(): 

    # set the model to inference mode 
    model.eval() 

    # Let's create a dummy input tensor  
    dummy_input = torch.randn(1, 3, 512, 512, requires_grad=True)  

    # Export the model   
    torch.onnx.export(model,         # model being run 
        dummy_input,       # model input (or a tuple for multiple inputs) 
        "HeartFailureClassifier.onnx",       # where to save the model  
        export_params=True,  # store the trained parameter weights inside the model file 
        opset_version=10,    # the ONNX version to export the model to 
        do_constant_folding=True,  # whether to execute constant folding for optimization 
        input_names = ["input"],   # the model's input names 
        output_names = ['output'], # the model's output names 
        dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                      'modelOutput' : {0 : 'batch_size'}}) 
    print(" ") 
    print('Model has been converted to ONNX') 


if __name__ == "__main__": 

    # Let's build our model 
    #train(5) 
    #print('Finished Training') 

    # Test which classes performed well 
    #testAccuracy() 

    # "cuda" only when GPUs are available.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize a model, and put it on the device specified.
    model = torch.load("220126_3_inceptionv2_resnet_model.pt", map_location='cpu')
    model.device = device

    # Make sure the model is in eval mode.
    # Some modules like Dropout or BatchNorm affect if the model is in training mode.
    model.eval()

    # Test with batch of images 
    #testBatch() 
    # Test how the classes performed 
    #testClassess() 
 
    # Conversion to ONNX 
    Convert_ONNX()