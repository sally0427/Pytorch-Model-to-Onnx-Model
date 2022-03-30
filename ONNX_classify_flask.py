import onnxruntime
import cv2
import numpy as np
import torchvision.transforms as transforms
from flask import Flask, request

ort_session = onnxruntime.InferenceSession("HeartFailureClassifier2.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    image_path = request.args.get('path')
    # img = cv2.imread("./images/428/FO-173140964955286438.png")
    # img = cv2.imread("./images/428/FO-33744932543442714.png")
    img = cv2.imread("./images/428/Normal-2.png")
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    to_tensor = transforms.ToTensor()
    img_y = to_tensor(img)
    img_y.unsqueeze_(0)

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_y)}
    ort_outs = ort_session.run(None, ort_inputs)
    img_out_y = ort_outs[0]

    index_max = np.argmax(img_out_y)
    if index_max == 0:
        print("428")
        logs = "Heart Failure"
    elif index_max == 1:
        print("n_428")
        logs = "Non Heart Failure"
    print("image path:", image_path)  
    
    return "Result : " + str(logs)

    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=False)
    # app.run(debug=False)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")