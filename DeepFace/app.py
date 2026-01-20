import streamlit as st
import cv2
import numpy as np
import time

import web_f
import web_mf
import web_mp


# ================= CONFIG =================
st.set_page_config(layout="wide")
st.title("Webcam feeds + Screenshot analyse")


# ================= SESSION =================
if "mode" not in st.session_state:
    st.session_state.mode = "LIVE"

if "screenshot" not in st.session_state:
    st.session_state.screenshot = None


# ================= MODE SWITCH =================
col_a, col_b = st.columns(2)

if col_a.button("LIVE MODE"):
    st.session_state.mode = "LIVE"

if col_b.button("ANALYSE MODE"):
    st.session_state.mode = "ANALYSE"


st.divider()


# =========================================================
# ========================= LIVE MODE =====================
# =========================================================
if st.session_state.mode == "LIVE":

    st.subheader("Live webcam feeds (stabil)")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Webcam kunne ikke åbnes")
        st.stop()

    col1, col2, col3 = st.columns(3)
    feed1 = col1.empty()
    feed2 = col2.empty()
    feed3 = col3.empty()

    btn_shot = st.button("Tag screenshot")

    frame_id = 0
    last_f1 = last_f2 = last_f3 = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        frame_id += 1

        # ML throttling
        if frame_id % 5 == 0 or last_f1 is None:
            small = cv2.resize(frame, (320, 240))

            last_f1 = cv2.resize(web_f.process_frame(small.copy()), (640, 480))
            last_f2 = cv2.resize(web_mf.process_frame(small.copy()), (640, 480))
            last_f3 = cv2.resize(web_mp.process_frame(small.copy()), (640, 480))

        feed1.image(cv2.cvtColor(last_f1, cv2.COLOR_BGR2RGB), caption="web_f")
        feed2.image(cv2.cvtColor(last_f2, cv2.COLOR_BGR2RGB), caption="web_mf")
        feed3.image(cv2.cvtColor(last_f3, cv2.COLOR_BGR2RGB), caption="web_mp")

        if btn_shot:
            st.session_state.screenshot = frame.copy()
            cap.release()
            st.session_state.mode = "ANALYSE"
            break

        time.sleep(0.01)


# =========================================================
# ====================== ANALYSE MODE =====================
# =========================================================
else:
    st.subheader("Screenshot analyse (ingen webcam)")

    if st.session_state.screenshot is None:
        st.warning("Intet screenshot taget")
        st.stop()

    src = st.session_state.screenshot

    # -------- SIDEBAR (FRI LEG – SIKKERT) --------
    with st.sidebar:
        mode = st.radio(
            "Filter",
            ["Ingen", "HSV", "Blob", "Canny"]
        )

        if mode == "HSV":
            h_min = st.slider("H min", 0, 179, 0)
            h_max = st.slider("H max", 0, 179, 179)
            s_min = st.slider("S min", 0, 255, 40)
            s_max = st.slider("S max", 0, 255, 255)
            v_min = st.slider("V min", 0, 255, 40)
            v_max = st.slider("V max", 0, 255, 255)

        if mode == "Blob":
            min_area = st.slider("Min area", 50, 5000, 300)

        if mode == "Canny":
            t1 = st.slider("T1", 0, 300, 100)
            t2 = st.slider("T2", 0, 300, 200)

    # -------- FILTER LOGIK --------
    out = src.copy()

    if mode == "HSV":
        hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array([h_min, s_min, v_min]),
            np.array([h_max, s_max, v_max])
        )
        out = cv2.bitwise_and(src, src, mask=mask)

    elif mode == "Blob":
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = min_area
        detector = cv2.SimpleBlobDetector_create(params)
        kp = detector.detect(gray)
        out = cv2.drawKeypoints(
            src, kp, None,
            (0, 0, 255),
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

    elif mode == "Canny":
        edges = cv2.Canny(
            cv2.cvtColor(src, cv2.COLOR_BGR2GRAY),
            t1, t2
        )
        out = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # -------- VISNING --------
    c1, c2 = st.columns(2)

    c1.image(cv2.cvtColor(src, cv2.COLOR_BGR2RGB), caption="Screenshot")
    c2.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), caption=mode)
