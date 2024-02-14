import streamlit as st
import os
import pickle
import tensorflow as tf
import cv2
import numpy as np
import tensorflow_model_optimization as tfmot

label_list = ["NonViolence","Violence"]

# Function to load the pre-trained model
@st.cache_resource
def load_model():
    # model = tf.keras.models.load_model('MoBiLSTM.h5')
    # return model
    with tfmot.quantization.keras.quantize_scope():
        loaded_model = tf.keras.models.load_model('MoBiLSTM.h5')
        return loaded_model


def save_video_locally(uploaded_file, custom_file_name):
    # Specify the file name with the desired extension
    file_extension = uploaded_file.name.split(".")[-1]
    saved_file_name = f"{custom_file_name}.{file_extension}"
    # Save the video to a temporary file on disk
    with open(saved_file_name, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return saved_file_name


def main():
    st.title("PeaceGuard: Violence Detection Platform")
    model = load_model()
    
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mkv"])
    
    if uploaded_file is None:
        st.warning("Please upload MP4 files only", icon="⚠️")
    

    if uploaded_file is not None:
        
        # Save the video file locally
        # save_button = st.button("Predict")

        saved_file_name = save_video_locally(uploaded_file, "input_video")
        # video_file = open('Violence Detection/input_video.mp4', 'rb')
        # video_bytes = video_file.read()
        # st.video(video_bytes)
        try:
            st.video(f"Violence Detection/{saved_file_name}")
        except Exception as e:
            print(f"Error: {e}")
            # Print the full traceback for more details
            import traceback
            traceback.print_exc()

        # After saving the video, display the results on a new page
        predict_video("/Users/anasshaik/Documents/Design Project/Violence Detection/input_video.mp4", model)

def save_video(uploaded_file):
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Video '{uploaded_file.name}' saved locally!")

def display_results_page():
    # Load the pre-trained model
    
    # Display the results on a new page
    st.subheader("Predicted Results")
    st.write("Replace this with your predicted result display logic")
    # st.write(f"Predicted Result: {predicted_result}")
    
def predict_video(video_file_path, model, SEQUENCE_LENGTH = 16):
 
    video_reader = cv2.VideoCapture(video_file_path)
 
    # Get the width and height of the video.
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
 
    # Declare a list to store video frames we will extract.
    frames_list = []
    
    # Store the predicted class in the video.
    predicted_class_name = ''
 
    # Get the number of frames in the video.
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
 
    # Calculate the interval after which frames will be added to the list.
    skip_frames_window = max(int(video_frames_count/16),1)
 
    # Iterating the number of times equal to the fixed length of sequence.
    for frame_counter in range(16):
 
        # Set the current frame position of the video.
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
 
        success, frame = video_reader.read() 
 
        if not success:
            break
 
        # Resize the Frame to fixed Dimensions.
        resized_frame = cv2.resize(frame, (64, 64))
        
        # Normalize the resized frame.
        normalized_frame = resized_frame / 255
        
        # Appending the pre-processed frame into the frames list
        frames_list.append(normalized_frame)
 
    # Passing the  pre-processed frames to the model and get the predicted probabilities.
    predicted_labels_probabilities = model.predict(np.expand_dims(frames_list, axis = 0))[0]
 
    # Get the index of class with highest probability.
    predicted_label = np.argmax(predicted_labels_probabilities)
 
    # Get the class name using the retrieved index.
    predicted_class_name = label_list[predicted_label]
    
    percentage = predicted_labels_probabilities[predicted_label] * 100 
    
    # Display the predicted class along with the prediction confidence.
    print(f'Predicted: {predicted_class_name}\nConfidence: {percentage:.3f} %')
    
    st.subheader(f'Predicted: {predicted_class_name}')
    
    st.video("/Users/anasshaik/Documents/Design Project/Violence Detection/input_video.mp4")
    
    st.subheader(f'Confidence: {percentage:.3f} %')
    
    video_reader.release()

if __name__ == "__main__":
    main()
