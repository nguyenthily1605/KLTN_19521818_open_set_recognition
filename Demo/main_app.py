import streamlit as st
import datetime
import importlib
import torch
import cv2

import numpy as np
from io import BytesIO,StringIO,open
from PIL import Image
from torch import nn
from Loss.ARPLoss import ARPLoss
from classifier32 import classifier32
from torchvision import transforms
from torchvision.models import mobilenetv3
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
parser.add_argument('--feat_dim', type=int, default=576, help="Feature vector dim, only for classifier32 at the moment")

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

def load_model(path_file_model='',flag=True,loss='Softmax'):
  if flag==True:
    model=classifier32(num_classes=4,feat_dim=576)
  else:
    model=mobilenetv3.mobilenet_v3_small(input=(32,32,3),num_classes=4)
    if  loss=='ARPLoss':
              model.classifier = nn.Sequential(*list(model.classifier.children())[:-4])
  model = nn.DataParallel(model).cpu()
  pretrain=torch.load(path_file_model,map_location=torch.device('cpu'))
  model.load_state_dict(pretrain)
  model.eval()
  return model
def load_image(image_file):
    img = Image.open(image_file)
    img=img.resize((350,205))
    return img
# Tieu de
def sosanh(xacsuat,threshold,predictions):
  new_title = '<p style="font-family:sans-serif; color:Red; font-size: 42px;">Kết quả</p>'
  st.markdown(new_title, unsafe_allow_html=True)
  col1,col2=st.columns(2)
  if xacsuat>threshold:
            if predictions.item()==2:
                st.success('Đối tượng thuộc phạm vi đã được học :  leopards', icon="✅")
            elif predictions.item()==1:
                st.success('Đối tượng thuộc phạm vi đã được học :  motorbikes', icon="✅")
            elif predictions.item()==0:
                st.success('Đối tượng thuộc phạm vi đã được học : airplanes', icon="✅")
                st.write(xacsuat)
            else:
                st.success('Đối tượng thuộc phạm vi đã được học : watch', icon="✅")
          
     
  else :
           st.error('Đối tượng không thuộc phạm vi được học', icon="🚨")
def Minh_hoa(uploaded_files,threshold,model,choice_pp="MSP",type_model="VGG32"):
        data=load_image(uploaded_files)
        data=test_transform(data)
        data1=data.unsqueeze(0)
        if type_model=="VGG32":
          x, y = model(data1, True)
        else: 
          x = model(data1)
        logits=x
        if choice_pp=="MSP":
            logits = torch.nn.Softmax(dim=-1)(logits)
            predictions_msp = logits.data.max(1)[1]
            xacsuat_msp=logits.data.max(1)[0].item()
            sosanh(xacsuat_msp,threshold,predictions_msp)
        
        #MLS
        elif choice_pp=="MLS":
            logits_mls = torch.nn.Softmax(dim=-1)(logits)  
            predictions_mls = logits.data.max(1)[1]
            xacsuat_mls=logits.data.max(1)[0].item()
            sosanh(xacsuat_mls,threshold,predictions_mls)
        
        #ARPL
        elif choice_pp=="ARPL":
        
          logits_arp, _ = criterion(x,x)
          logits_arp = torch.nn.Softmax(dim=-1)(logits_arp)
          predictions_arpl = logits_arp.data.max(1)[1]
          xacsuat_arpl=logits_arp.data.max(1)[0].item()
          sosanh(xacsuat_arpl,threshold,predictions_arpl)
 
new_title = '<p style="font-family:sans-serif; color:Red; font-size: 42px;">Sử dụng mạng học sâu cho nhận diện không gian mở</p>'
st.markdown(new_title, unsafe_allow_html=True)
# Select photo a send it to button
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.horizontal = True
with st.sidebar:
    title_menu = '<p style="font-family:sans-serif; color:Black; font-size: 35px;"> Mô hình</p>'
    st.markdown(title_menu,unsafe_allow_html=True)
    choice_mohinh=st.radio("",("   VGG32","    Mobilenetv3"),index=1)
    title_menu = '<p style="font-family:sans-serif; color:Black; font-size: 35px;"> Phương pháp </p>'
    st.markdown(title_menu,unsafe_allow_html=True)
    choice_pp=st.radio(
        "",
        ["MSP", "MLS", "ARPL"],index=2,
        key="visibility",
        horizontal=st.session_state.horizontal,
    )

st.sidebar.subheader("Chọn ngưỡng")
msp=st.sidebar.slider("",0.0,1.0) 
title_menu = '<p style="font-family:sans-serif; color:Black; font-size: 30px;"> Upload ảnh </p>' 
st.markdown(title_menu,unsafe_allow_html=True)
uploaded_files1= st.file_uploader('',type=['jpg','png'])
Loss = importlib.import_module('Loss.ARPLoss')
criterion = getattr(Loss, options['loss'])(**options)
if(uploaded_files1 is not None):
  st.image(load_image(uploaded_files1),channels = 'BGR',use_column_width=True)
  if choice_mohinh=='   VGG32': 
    if choice_pp=="MSP" or choice_pp=="MLS":
            vid_known=''
            model=load_model(path_file_model='vgg32_caltech_msp_model')
            Minh_hoa(uploaded_files=uploaded_files1,threshold=msp,model=model,choice_pp=choice_pp)
    else:
                       #criterion = ARPLoss(options)
            criterion = criterion.cpu()
            criterion.load_state_dict(torch.load('vgg32_caltech_arpl_criterion',map_location=torch.device('cpu')))
            criterion.eval()
            model=load_model(path_file_model='vgg32_caltech_arpl_model')
            Minh_hoa(uploaded_files=uploaded_files1,threshold=msp,model=model,choice_pp=choice_pp)
  else:
    if choice_pp=="MSP" or choice_pp=="MLS":
            vid_known=''
            model1=load_model(path_file_model='modilenetv3_caltech_msp_model',flag=False)
            Minh_hoa(uploaded_files=uploaded_files1,threshold=msp,model=model1,choice_pp=choice_pp,type_model="Mobilenetv3")
    else:
                       #criterion = ARPLoss(options)
            criterion = criterion.cpu()
            criterion.load_state_dict(torch.load('mobile_caltech_criterion_arpl',map_location=torch.device('cpu')))
            criterion.eval()
            model=load_model(path_file_model='mobile_caltech_model_arpl',flag=False,loss='ARPLoss')
            Minh_hoa(uploaded_files=uploaded_files1,threshold=msp,model=model,choice_pp=choice_pp,type_model="Mobilenetv3")
  
    
            
      

        
   

    
        


        
        

   
    
    

    
