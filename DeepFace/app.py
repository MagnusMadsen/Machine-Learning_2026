import streamlit as st
import cv2
import numpy as np

import verify_face
import verify_webcam_multiface
import verify_webcam_multiplayer

if "screenshot" not in st.session_state:
    st.session_state["screenshot"] = None

st.set_page_config(layout="wide")
st.title("Face Recognition")

# webcam capture
cap = cv2.VideoCapture(0)

# layout
col1, col2, col3 = st.columns(3)
feed1 = col1.empty()
feed2 = col2.empty()
feed3 = col3.empty()

st.divider()
st.subheader("Screenshot & billedanalyse")

btn = st.button("Tag screenshot")

a1, a2, a3, a4 = st.columns(4)
orig_view = a1.empty()
blur_view = a2.empty()
hsv_view  = a3.empty()
gray_view = a4.empty()

last_frame = None

# --- LOOP ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    last_frame = frame.copy()

    f1 = verify_face.process_frame(frame.copy())
    f2 = verify_webcam_multiface.process_frame(frame.copy())
    f3 = verify_webcam_multiplayer.process_frame(frame.copy())

    feed1.image(cv2.cvtColor(f1, cv2.COLOR_BGR2RGB), caption="Daniel vs Others")
    feed2.image(cv2.cvtColor(f2, cv2.COLOR_BGR2RGB), caption="Multi-face")
    feed3.image(cv2.cvtColor(f3, cv2.COLOR_BGR2RGB), caption="Multi-player")

    if btn and last_frame is not None:
        blur = cv2.GaussianBlur(last_frame, (15,15), 0)
        hsv  = cv2.cvtColor(last_frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)

        orig_view.image(cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB), caption="Original")
        blur_view.image(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB), caption="Blur")
        hsv_view.image(hsv, caption="HSV")
        gray_view.image(gray, caption="Gray")



        st.subheader("HSV kontrolpanel")

if hsv_view in last_frame:
    bgr = last_frame["screenshot"]
    bgr = cv2.resize(bgr, (400, 600))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # --- SLIDERS  ---
    col1, col2 = st.columns(2)

    with col1:
        h_min = st.slider("H min", 0, 179, 0)
        s_min = st.slider("S min", 0, 255, 40)
        v_min = st.slider("V min", 0, 255, 40)

    with col2:
        h_max = st.slider("H max", 0, 179, 25)
        s_max = st.slider("S max", 0, 255, 255)
        v_max = st.slider("V max", 0, 255, 230)

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
