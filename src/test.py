import numpy as np
from PIL import Image
import cv2

from seg_model import SegModel



def load_model(path: str = '../models/trained_model_unet++.pth') -> SegModel:
    """Retrieves the trained model and maps it to the CPU by default,
    can also specify GPU here."""
    model = SegModel(path_to_pretrained_model=path)
    return model



def predict(
        img: Image.Image,
        model
        ):
    """Transforming inputs image according to ImageNet paper
    The Resnet was initially trained on ImageNet dataset
    and because of the use of transfer learning, I froze all
    weights and only learned weights on the final layer.
    The weights of the first layer are still what was
    used in the ImageNet paper and we need to process
    the new images just like they did.

    This function transforms the image accordingly,
    puts it to the necessary device (cpu by default here),
    feeds the image through the model getting the output tensor,
    converts that output tensor to probabilities using Softmax,
    and then extracts and formats the top k predictions."""
    pre = model.predict_proba(img)
    return pre


if __name__ == '__main__':
    model = load_model()#加载模型

    img =cv2.imread("./1.png")
    print("88888888888888888", img)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    prediction = predict(img, model)
    prediction.show()
    print("************", prediction)