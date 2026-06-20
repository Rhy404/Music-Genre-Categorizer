import streamlit as st
import joblib
import os
from utils import extract_audio_features

model = joblib.load("music_genre_xgb.pkl")
le = joblib.load("label_encoder.pkl")

st.set_page_config(page_title="Music Genre Categorizer", page_icon="♾️")

st.title("Music Genre Categorizer")
st.write("Upload an MP3 file to identify its genre using the XGBoost model.")

uploaded_file = st.file_uploader("Choose an MP3 file", type=["mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/mp3")
    
    if st.button("Identify Genre"):
        with st.spinner("Processing audio...Listen to your fav song and wait for a moment"):
            temp_filename = "temp_upload.mp3"
            with open(temp_filename, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            try:
                features_df = extract_audio_features(temp_filename)
                
                prediction_numeric = model.predict(features_df)
                genre_name = le.inverse_transform(prediction_numeric)[0]
                
                st.success(f"The predicted genre is: **{genre_name.upper()}**")
            
            except Exception as e:
                st.error(f"An error occurred: {e}")
                
            finally:
                # clean up the temporary file
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
            