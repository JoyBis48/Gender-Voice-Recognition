# type: ignore

# Importing the required libraries
import io
import streamlit as st 
import numpy as np
import librosa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from scipy.io.wavfile import write, read as wav_read 
# import sounddevice as sd    
from st_audiorec import st_audiorec


# Function to record live voice and save as WAV
# def record_audio(duration, sr):
#     st.write("Recording...")
#     audio_file = "recorded_audio.wav"
#     recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
#     sd.wait()
#     write(audio_file, sr, recording)
#     return audio_file


# Function to convert audio to spectrogram image
def audio_to_spectrogram(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=512)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    plt.figure(figsize=(4, 4))
    plt.axis('off')
    plt.imshow(mel_spec_db, aspect='auto', origin='lower')
    plt.tight_layout()
    plt.savefig("spectrogram.png")
    plt.close()

# Function to create the gender classification model
def create_model(vector_length=128):
    model = Sequential([
    Dense(256, input_shape=(vector_length,), activation='relu'),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')
    model.summary()
    return model

# Load the pre-trained model
model = create_model()
model.load_weights("saved_model.h5")

# Streamlit app
st.title("Voice Gender Detection")
st.write("This application detects the gender from recorded voice using a n.")

# Option to upload a file
uploaded_file = st.file_uploader("Upload an audio file", type=['wav', 'mp3'])

# Function to extract features from audio file
def extract_feature(file_name):
    X, sample_rate = librosa.core.load(file_name)
    result = np.array([])
    mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
    result = np.hstack((result, mel))
    return result

# Function to classify gender
def classify_gender(file_path):
    features = extract_feature(file_path).reshape(1, -1)
    male_prob = model.predict(features, verbose=0)[0][0]
    female_prob = 1 - male_prob
    gender = "male" if male_prob > female_prob else "female"
    probability = round(male_prob, 2) if gender == "male" else round(female_prob, 2)
    return gender, probability

if uploaded_file is not None:
    with open("uploaded_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.audio(uploaded_file, format='audio/wav')
    if st.button("Submit"):
        audio_to_spectrogram("uploaded_audio.wav")
        st.image("spectrogram.png", caption="Mel Spectrogram of the uploaded audio file", use_column_width="auto", width=200)
        gender, probability = classify_gender("uploaded_audio.wav")
        st.write(f"Predicted Gender: {gender}")
        st.write(f"Probability: {probability}")

# if st.sidebar.button("Start Recording"):
#     audio_file = record_audio(duration, sr=22050)
#     st.audio(audio_file, format="audio/wav")
#     audio_to_spectrogram(audio_file)
#     st.image("spectrogram.png", caption="Mel Spectrogram of the uploaded audio file", use_column_width="auto", width=200)
#     gender, probability = classify_gender(audio_file)
#     st.write(f"Predicted Gender: {gender}")
#     st.write(f"Probability: {probability}")

wav_audio_data = st_audiorec()


if wav_audio_data is not None:
    # Convert byte string to numpy array
    wav_io = io.BytesIO(wav_audio_data)
    sr, audio_data = wav_read(wav_io)

    # Save numpy array to WAV file
    wav_file_path = "recorded_audio.wav"
    write(wav_file_path, sr, audio_data)

    st.audio(wav_audio_data, format='audio/wav')
    audio_to_spectrogram(wav_file_path)
    st.image("spectrogram.png", caption="Mel Spectrogram of the uploaded audio file", use_column_width="auto", width=200)
    gender, probability = classify_gender(wav_file_path)
    st.write(f"Predicted Gender: {gender}")
    st.write(f"Probability: {probability}")
