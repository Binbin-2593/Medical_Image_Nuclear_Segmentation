from random import random

import cv2
import numpy as np
from PIL import Image
import streamlit as st


from seg_model import SegModel


@st.cache_data()#通过ResnetModel类加载模型
def load_model(path: str = 'models/trained_model_unet++.pth') -> SegModel:
    """Retrieves the trained model and maps it to the CPU by default,
    can also specify GPU here."""
    model = SegModel(path_to_pretrained_model=path)
    return model


@st.cache_data()#输入图像，模型预测
def predict(
        img: Image.Image,
        _model
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
    model = load_model()  # 加载模型

    st.title('Welcome To Project Image Segmentation Of Nucleus!')
    instructions = """
        Either upload your own image or select from
        the sidebar to get a preconfigured image.
        The image you select or upload will be fed
        through the Deep Neural Network in real-time
        and the output will be displayed to the screen.
        """
    st.write(instructions)

    file = st.file_uploader('Upload An Image')

    if file:  # 如果用户上传文件

        img = Image.open(file)

        st.title("Here is the image you've selected")
        st.image(img)

        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

        prediction = predict(img, model)
        st.title("The following is the segmented prediction image")
        st.image(prediction)



































