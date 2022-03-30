import onnxruntime
import cv2
import numpy as np
import torchvision.transforms as transforms

# ort_session = onnxruntime.InferenceSession("HeartFailureClassifier2.onnx")
ort_session = onnxruntime.InferenceSession("model-f6b98070.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


img = cv2.imread("./images/428/FO-173140964955286438.png")
img = cv2.imread("./images/428/FO-33744932543442714.png")
# img = cv2.imread("./images/428/Normal-2.png")
print(img.shape)
img = cv2.resize(img, (384, 384), interpolation=cv2.INTER_AREA)
# np_arr = np.expand_di
# img = np.moveaxis(img, -1, 0)
print(img.shape)
# np_arr = np.expand_dims(img, axis=0)
to_tensor = transforms.ToTensor()
img_y = to_tensor(img)
img_y.unsqueeze_(0)

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_y)}
ort_outs = ort_session.run(None, ort_inputs)
img_out_y = ort_outs[0]

index_max = np.argmax(img_out_y)
# print(type(to_numpy(img_y)))
# print(index_max)

# if index_max == 0:
#     print("428")
# elif index_max == 1:
#     print("n_428")

# img_out_y = Image.fromarray(np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]), mode='L')
img_out_y.argmax(dim=-1).numpy().tolist()

# compare ONNX Runtime and PyTorch results
# np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
print("Exported model has been tested with ONNXRuntime, and the result looks good!")



# print(type(ort_inputs))
# print(np.shape(ort_inputs['modelInput']))
# print(type(ort_inputs['modelInput']))
# print(ort_inputs['modelInput'])
# print(ort_session.get_inputs()[0].name)
# print(ort_session.get_outputs()[0].name)
# print(type(ort_outs))
# print(np.shape(ort_outs))
# print(ort_outs)