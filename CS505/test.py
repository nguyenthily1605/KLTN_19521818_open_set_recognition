import streamlit as st
import os
import time
import cv2
import tempfile

demo='unknown.mp4'
stframe = st.empty()
vid = cv2.VideoCapture(demo)

fps = 0
i = 0
while vid.isOpened():
            i +=1
            ret, frame = vid.read()
            st.write(str(frame.shape))
            col1,col2=st.columns(2)
            with col1:
                if ret:
                    stframe.image(frame,channels = 'BGR',use_column_width=True)
                else:
                    vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
            with col2: 
                st.write(str(frame.shape))

vid.release()

