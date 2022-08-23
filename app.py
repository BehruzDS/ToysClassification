import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import pathlib
import platform
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

# title
st.title("O`yinchoqlarni klassifikatsiya qiladigan model")

# rasm yuklash
file = st.file_uploader("Rasm yuklash", type=['png','jpeg','jpg','gif','jfif','pjp','svg'])

if file:
    # PILImage
    img = PILImage.create(file)

    # modelni chaqirib olamiz
    model = load_learner('toys_model.pkl')

    # rasm
    st.image(file)
    # bashorat
    pred, pred_id, probs = model.predict(img)

    # chop qilish
    st.success(f'Bashorat: {pred}')
    st.info(f'Ehtimollik: {probs[pred_id]*100:1f} %')

    fig = px.bar(x=probs*100, y= model.dls.vocab)
    st.plotly_chart(fig)
