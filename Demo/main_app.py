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

#MSP
data_msp=pd.read_csv('https://github.com/nguyenthily1605/KLTN_19521818_open_set_recognition/blob/main/Demo/MLS_cifar10.csv' ,error_bad_lines=False)
fig_1_msp, ax_1_msp = plt.subplots()
fig_2_msp, ax_2_msp = plt.subplots()
fig_3_msp, ax_3_msp = plt.subplots()
ax_1_msp.set_xlabel('Epoch')
ax_1_msp.set_ylabel('Acuracy(Closed Set Performance)')
ax_1_msp.plot( 'epoch', 'acuracy', data=data_msp)
ax_2_msp.set_xlabel('Epoch')
ax_2_msp.set_ylabel('AUROC (Open-set Performance)')
ax_2_msp.plot( 'epoch', 'auroc', data=data_msp)
ax_3_msp.set_xlabel('Acuracy(Closed Set Performance)')
ax_3_msp.set_ylabel('AUROC (Open-set Performance)')
ax_3_msp.plot( 'acuracy', 'auroc', data=data_msp, linestyle='none', marker='o')

#ARPL
data_arpl=pd.read_csv('.\ARPL_cifar10.csv')
fig_1_arpl, ax_1_arpl = plt.subplots()
fig_2_arpl, ax_2_arpl = plt.subplots()
fig_3_arpl, ax_3_arpl = plt.subplots()
ax_1_arpl.set_xlabel('Epoch')
ax_1_arpl.set_ylabel('Acuracy(Closed Set Performance)')
ax_1_arpl.plot( 'epoch', 'acuracy', data=data_arpl)
ax_2_arpl.set_xlabel('Epoch')
ax_2_arpl.set_ylabel('AUROC (Open-set Performance)')
ax_2_arpl.plot( 'epoch', 'auroc', data=data_arpl)
ax_3_arpl.set_xlabel('Acuracy(Closed Set Performance)')
ax_3_arpl.set_ylabel('AUROC (Open-set Performance)')
ax_3_arpl.plot( 'acuracy', 'auroc', data=data_arpl, linestyle='none', marker='o')

#MLS
data_mls=pd.read_csv('.\MLS_cifar10.csv')
fig_1_mls, ax_1_mls = plt.subplots()
fig_2_mls, ax_2_mls = plt.subplots()
fig_3_mls, ax_3_mls = plt.subplots()
ax_1_mls.set_xlabel('Epoch')
ax_1_mls.set_ylabel('Acuracy(Closed Set Performance)')
ax_1_mls.plot( 'epoch', 'acuracy', data=data_mls)
ax_2_mls.set_xlabel('Epoch')
ax_2_mls.set_ylabel('AUROC (Open-set Performance)')
ax_2_mls.plot( 'epoch', 'auroc', data=data_mls)
ax_3_mls.set_xlabel('Acuracy(Closed Set Performance)')
ax_3_mls.set_ylabel('AUROC (Open-set Performance)')
ax_3_mls.plot( 'acuracy', 'auroc', data=data_mls, linestyle='none', marker='o')

test_transform = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
def load_model(path_file_model=''):
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
#MSP AND MLS
model_msp=load_model(path_file_model='.\weights_cifar.pth')
model_msp.eval()

#ARPL
model_arpl=load_model('.\ARPL.pth')
model_arpl.eval()
Loss = importlib.import_module('Loss.'+options['loss'])
criterion = getattr(Loss, options['loss'])(**options)
criterion=criterion.cpu()
criterion.load_state_dict(torch.load('.\ARPL_loss.pth',map_location=torch.device('cpu')))


# Tieu de
def Minh_hoa(uploaded_files,threshold_msp):
    xacsuat_msp=0
    xacsuat_mls=0
    xacsuat_arp=0
    if uploaded_files is not None:

        data=load_image(uploaded_files)
        data=test_transform(data)
        data1=data.unsqueeze(0)
        x_msp, y_msp = model_msp(data1, True)
        logits_msp=y_msp
        logits_msp = torch.nn.Softmax(dim=-1)(logits_msp)
        predictions_msp = logits_msp.data.max(1)[1]
        xacsuat_msp=logits_msp.data.max(1)[0].item()
        
        #MLS
        x_mls, y_mls = model_msp(data1, True)
        logits_mls=y_mls
    #logits_mls = torch.nn.Softmax(dim=-1)(logits_mls)  
        predictions_mls = logits_mls.data.max(1)[1]
        xacsuat_mls=logits_mls.data.max(1)[0].item()
        
        #ARPL
        x_arp, y_arp = model_arpl(data1, True)
        logits_arp=y_arp
        logits_arp, _ = criterion(x_arp, y_arp)
        logits_arp = torch.nn.Softmax(dim=-1)(logits_arp)
        predictions_arp = logits_arp.data.max(1)[1]
        xacsuat_arp=logits_arp.data.max(1)[0].item()
        
        new_title = '<p style="font-family:sans-serif; color:Red; font-size: 42px;">Kết quả</p>'
        st.markdown(new_title, unsafe_allow_html=True)
    #st.header("Kết quả")
        kq=[xacsuat_msp,xacsuat_arp]
        kq=np.array(kq)
        col1,col2=st.columns(2)
   
        if xacsuat_arp>threshold_msp:
            if predictions_arp.item()==0:
                vid_known='deer.mp4'
            elif predictions_arp.item()==1:
                vid_known='horse.mp4'
            elif predictions_arp.item()==2:
                vid_known='truck.mp4'
            else:
                vid_known='auto.mp4'
            with col2:
                st.image(load_image(img_unknown),channels = 'BGR',use_column_width=True)
             
            with col1:
               
                load_vid(vid_known)

        else:
            with col1:
                st.image(load_image(img_known),channels = 'BGR',use_column_width=True)
            with col2:
                load_vid(vid_unknown)
 
new_title = '<p style="font-family:sans-serif; color:Red; font-size: 42px;">Sử dụng mạng học sâu cho nhận diện không gian mở</p>'
st.markdown(new_title, unsafe_allow_html=True)
# Select photo a send it to button
with st.sidebar:
    title_menu = '<p style="font-family:sans-serif; color:Black; font-size: 42px;"> 🏠 Menu</p>'
    st.markdown(title_menu,unsafe_allow_html=True)
    choice=st.radio("",(" 	🖌️  Minh họa"," 📈 Thống kê"))
    st.subheader("Chọn ngưỡng")
    msp=st.slider("",0.0,1.0)  
test=0
uploaded_files=st.empty()
st.write()
img_known='known.jpg'
img_unknown='unknown.jpg'
vid_unknown='unknown.mp4'
if choice==' 	🖌️  Minh họa': 
        uploaded_files1= st.file_uploader(" Upload ảnh ",type=['jpg','png'])
        if(uploaded_files1 is None):
            col1,col2=st.columns(2)
            with col1:
                st.image(load_image('known.jpg'))
            with col2:
                st.image(load_image('unknown.jpg'))
        if(uploaded_files1 is not None):
            max=0
            index_max=0
            vid_known=''
            img=st.image(load_image(uploaded_files1),channels = 'BGR',use_column_width=True)
            Minh_hoa(uploaded_files=uploaded_files1,threshold_msp=msp)

        
   
elif choice==" 📈 Thống kê":
    st.subheader("Phương pháp MSP")
    st.write("          Bảng kết quả    ")
    data_msp=data_msp.drop('Unnamed: 0',axis=1)
    st.table(data_msp.head(10))
    st.pyplot(fig_1_msp)
    st.pyplot(fig_2_msp)
    st.pyplot(fig_3_msp)
    
    st.subheader("Phương pháp MLS")
    st.write("          Bảng kết quả    ")
    data_mls=data_mls.drop('Unnamed: 0',axis=1)
    st.table(data_mls.head(10))
    st.pyplot(fig_1_mls)
    st.pyplot(fig_2_mls)
    st.pyplot(fig_3_mls)
    
    
    st.subheader("Phương pháp ARPL")
    st.write("          Bảng kết quả    ")
    data_arpl=data_arpl.drop('Unnamed: 0',axis=1)
    st.table(data_arpl.head(10))
    st.pyplot(fig_1_arpl)
    st.pyplot(fig_2_arpl)
    st.pyplot(fig_3_arpl)
    
        


        
        

   
    
    

    
