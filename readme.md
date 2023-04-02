# Feelingo

Feelingo is a music recommendation system that analyzes the user's emotion from video and audio inputs and recommends a playlist of songs based on the detected emotion. The system is built using Python and leverages various libraries including DeepFace, OpenCV, NLTK, SpeechRecognition, Spotipy, and the KNN algorithm.

The system uses DeepFace to analyze the user's emotion from a video input and OpenCV to capture the user's face in real-time. It also uses SpeechRecognition to capture the user's speech and extract keywords that help determine the user's emotional state. Additionally, NLTK is used to perform sentiment analysis on the user's text inputs.

To recommend the playlist, the system utilizes the Spotipy library to search for songs and the KNN algorithm to select songs that match the user's emotional state. The recommended playlist is displayed using Streamlit, which makes it easy to use and accessible in a web browser.

# Requirements
```
-> Python 3.x
-> DeepFace
-> OpenCV
-> NLTK
-> SpeechRecognition
-> Spotipy
-> scikit-learn
-> Streamlit
```

# Installation and Usage
``` 
 Clone this repository
 Install the required libraries
 Open Terminal from the Directory "streamlit"
 Run the Streamlit app using streamlit run intro.py
 Copy & Paste the URL obtained after running the above command in a browser```
