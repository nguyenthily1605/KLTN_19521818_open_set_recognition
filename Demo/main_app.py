import streamlit as st
import datetime
import importlib
import torch
import cv2

import numpy as np
from io import BytesIO,StringIO,open
from PIL import Image
from torch import nn
from classifier32 import classifier32
from torchvision import transforms
import argparse
import pandas as pd
l=cv2.__version__
import matplotlib.pyplot as plt
from datetime import datetime
parser = argparse.ArgumentParser("Training")
parser.add_argument('-f')
parser.add_argument('--image_size', type=int, default=32)
# optimization
parser.add_argument('--temp', type=float, default=1.0, help="temp")
# model
parser.add_argument('--loss', type=str, default='ARPLoss')
parser.add_argument('--weight-pl', type=float, default=0.1, help="weight for center loss")
parser.add_argument('--label_smoothing', type=float, default=None, help="Smoothing constant for label smoothing."
                                                                        "No smoothing if None or 0")
parser.add_argument('--feat_dim', type=int, default=128, help="Feature vector dim, only for classifier32 at the moment")

args = parser.parse_args()
options = vars(args)
options.update(
            {
                'num_classes': 4
            }
        )

device = "cuda" if torch.cuda.is_available() else "cpu"
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)
name=['deer','horse','truck','automobile']
test_transform = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

def load_model(path_file_model='',flag=True):
  if flag==True:
    model=classifier32(num_classes=4)
  model = nn.DataParallel(model).cpu()
  pretrain=torch.load(path_file_model,map_location=torch.device('cpu'))
  model.load_state_dict(pretrain)
  return model
def load_image(image_file):
    img = Image.open(image_file)
    img=img.resize((350,205))
    return img
def load_vid(path):
    stframe = st.empty()
    vid = cv2.VideoCapture(path)
    while vid.isOpened():
                    ret, frame = vid.read()
                    if ret:
                        stframe.image(frame,channels = 'BGR',use_column_width=True)
                    else:
                        vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue

# Tieu de
def sosanh(xacsuat,threshold,predictions):
  col1,col2=st.columns(2)
  if xacsuat>threshold:
            if predictions.item()==0:
                vid_known='deer.mp4'
            elif predictions.item()==1:
                vid_known='horse.mp4'
            elif predictions.item()==2:
                vid_known='truck.mp4'
            else:
                vid_known='auto.mp4'
            with col2:
                st.image(load_image(img_unknown),channels = 'BGR',use_column_width=True)
             
            with col1:
               
                load_vid(vid_known)
     
  else :
            with col1:
                st.image(load_image(img_known),channels = 'BGR',use_column_width=True)
            with col2:
                load_vid(vid_unknown)
def Minh_hoa(uploaded_files,threshold,model,flag_msp=True,flag_mls=False,flag_arpl=False,type_model="VGG32"):
        data=load_image(uploaded_files)
        data=test_transform(data)
        data1=data.unsqueeze(0)
        new_title = '<p style="font-family:sans-serif; color:Red; font-size: 42px;">Kết quả</p>'
        st.markdown(new_title, unsafe_allow_html=True)
        if type_model=="VGG32":
          x, y = model(data1, True)
          logits=y
        if flag_msp==True:
          logits = torch.nn.Softmax(dim=-1)(logits)
          predictions_msp = logits.data.max(1)[1]
          xacsuat_msp=logits.data.max(1)[0].item()
          sosanh(xacsuat_msp,threshold,predictions_msp)
        
        #MLS
        if flag_mls==True:
    #logits_mls = torch.nn.Softmax(dim=-1)(logits_mls)  
          predictions_mls = logits.data.max(1)[1]
          xacsuat_mls=logits.data.max(1)[0].item()
          sosanh(xacsuat_mls,threshold,predictions_mls)
        
        #ARPL
        if flag_arpl==True:
            logits_arp, _ = criterion(x, y)
            logits_arp = torch.nn.Softmax(dim=-1)(logits_arp)
            predictions_arp = logits_arp.data.max(1)[1]
            xacsuat_arp=logits_arp.data.max(1)[0].item()
            sosanh(xacsuat_arpl,threshold,predictions_arpl)
        
        new_title = '<p style="font-family:sans-serif; color:Red; font-size: 42px;">Kết quả</p>'
        st.markdown(new_title, unsafe_allow_html=True)
        
 
new_title = '<p style="font-family:sans-serif; color:Red; font-size: 42px;">Sử dụng mạng học sâu cho nhận diện không gian mở</p>'
st.markdown(new_title, unsafe_allow_html=True)
# Select photo a send it to button
with st.sidebar:
    title_menu = '<p style="font-family:sans-serif; color:Black; font-size: 35px;"> 🏠 Mô hình</p>'
    st.markdown(title_menu,unsafe_allow_html=True)
    choice_mohinh=st.radio("",("   VGG32","    Mobilenetv3"))
    title_menu = '<p style="font-family:sans-serif; color:Black; font-size: 35px;"> Phương pháp </p>'
    st.markdown(title_menu,unsafe_allow_html=True)
def style_button_row(clicked_button_ix, n_buttons):
    def get_button_indices(button_ix):
        return {
            'nth_child': button_ix,
            'nth_last_child': n_buttons - button_ix + 1
        }

    clicked_style = """
    div[data-testid*="stHorizontalBlock"] > div:nth-child(%(nth_child)s):nth-last-child(%(nth_last_child)s) button {
        border-color: rgb(255, 75, 75);
        color: rgb(255, 75, 75);
        box-shadow: rgba(255, 75, 75, 0.5) 0px 0px 0px 0.2rem;
        outline: currentcolor none medium;
        font-size: 20px;
    }
    """
    unclicked_style = """
    div[data-testid*="stHorizontalBlock"] > div:nth-child(%(nth_child)s):nth-last-child(%(nth_last_child)s) button {
        pointer-events: none;
        cursor: not-allowed;
        opacity: 0.65;
        filter: alpha(opacity=65);
        -webkit-box-shadow: none;
        box-shadow: none;
    }
    """
    style = ""
    for ix in range(n_buttons):
        ix += 1
        if ix == clicked_button_ix:
            style += clicked_style % get_button_indices(ix)
    st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)
col1, col2, col3 = st.sidebar.columns([1, 1, 1])
with col1:
    flag_msp=st.button("MSP", on_click=style_button_row, kwargs={
        'clicked_button_ix': 1, 'n_buttons': 4
    })
with col2:
    flag_mls=st.button("MLS", on_click=style_button_row, kwargs={
        'clicked_button_ix': 2, 'n_buttons': 4
    })
with col3:
    flag_arpl=st.button("ARPL", on_click=style_button_row, kwargs={
       'clicked_button_ix': 3, 'n_buttons': 4

    })
st.sidebar.subheader("Chọn ngưỡng")
msp=st.sidebar.slider("",0.0,1.0) 
title_menu = '<p style="font-family:sans-serif; color:Black; font-size: 30px;"> Upload ảnh </p>' 
st.markdown(title_menu,unsafe_allow_html=True)
uploaded_files1= st.file_uploader('',type=['jpg','png'])
if(uploaded_files1 is None):
            col1,col2=st.columns(2)
            with col1:
                st.image(load_image('known.jpg'))
            with col2:
                st.image(load_image('unknown.jpg'))
img_known='known.jpg'
img_unknown='unknown.jpg'
vid_unknown='unknown.mp4'
if(uploaded_files1 is not None):
  #st.image(load_image(uploaded_files1),channels = 'BGR',use_column_width=True)
  if choice_mohinh=='   VGG32': 
    if flag_msp==True or flag_mls==True:
            vid_known=''
            model=load_model(path_file_model='weights_cifar.pth')
            st.image(load_image(uploaded_files1),channels = 'BGR',use_column_width=True)
            Minh_hoa(uploaded_files=uploaded_files1,threshold=msp,model=model,flag_msp=flag_msp,flag_mls=flag_mls,flag_arpl=flag_arpl)

        
   

    
        


        
        

   
    
    

    
