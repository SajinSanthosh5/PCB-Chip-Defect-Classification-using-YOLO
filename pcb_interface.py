import PIL.Image
import numpy as np
import streamlit as st
import PIL
from ultralytics import YOLO
st.title("PCB detection model")
model=YOLO("PCB_best.pt")
load=st.file_uploader("uplaod PCB image",type=["png","jpg","png"])
if load:
    image=PIL.Image.open(load)
    st.image(image,caption="image uploaded")
    img_arr=np.array(image)
    res=model(img_arr)
    annoted_frame=res[0].plot()
    st.image(annoted_frame,caption="image detected")