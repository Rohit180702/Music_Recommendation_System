import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
#nltk.download('punkt')
#nltk.download('vader_lexicon')
text = 'happy'
finalev = 'happy'
def get_text():
    import speech_recognition as sr

    r = sr.Recognizer()

    with sr.Microphone() as source:

        r.energy_threshold = 300
        r.dynamic_energy_threshold = True
        r.dynamic_energy_adjustment_ratio = 1.5
        r.pause_threshold = 0.5
        r.non_speaking_duration = 0.3
        r.operation_timeout = None
        r.phrase_threshold = 0.3
        r.max_silence = 1
        r.adjust_for_ambient_noise(source, duration=0.5)

        print("Speak...")
    
        try:
            audio = r.listen(source, phrase_time_limit=5)
    
            text = r.recognize_google(audio,language=  "EN_US")

            print("You said: " + text)

        except sr.UnknownValueError:

            print("Could not understand audio")

        except sr.RequestError as e:

            print("Error: {0}".format(e))
    return "i am happy"
    
sid = SentimentIntensityAnalyzer()
output = get_text()
word_tokenize(output)
scores = sid.polarity_scores(output)
if scores['compound'] >= 0.2 and scores['compound']<0.8:
    finalev = 'happy'
elif scores['compound'] <= -0.2 and scores['compound']>-0.75:
    finalev = 'sad'
elif scores['compound'] > -0.2 and scores['compound'] < 0.2:
    finalev = 'normal'
elif scores['compound'] <=-0.75:
    finalev = 'angry'
elif scores['compound']>0.8:
    finalev = 'surprise'
else:
    finalev = 'fear'


###########################333
import spotipy
import pandas as pd
import random
from sklearn.neighbors import NearestNeighbors
from spotipy.oauth2 import SpotifyClientCredentials
#Authentication - without user

def recommend_songs(emotion, num_to_select):
    num_songs=20
    # create a new array with the normalized features for the input emotion
    input_emotion = train.loc[train[f"emotion_{emotion}"] == 1].values[:, 0:]
    # find the nearest neighbors for the input emotion
    distances, indices = nn.kneighbors(input_emotion)
    # randomly select a subset of the top recommended songs
    top_songs = df1.iloc[indices[0]][['name', 'artist']].head(num_songs)
    selected_songs = top_songs.sample(n=num_to_select, replace=False)
    return selected_songs
client_credentials_manager = SpotifyClientCredentials(client_id="f99b4dafd3504ab389e943951d0efa39", client_secret="13b6230cb77e4f46a8d2eabb27292750")
sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)
playlist_link = "https://open.spotify.com/playlist/78Sdax2ygDKQiVtbUtzjWQ"
playlist_URI = playlist_link.split("/")[-1].split("?")[0]
track_uris = [x["track"]["uri"] for x in sp.playlist_tracks(playlist_URI)["items"]]
# Retrieve the list of tracks in the playlist
results = sp.playlist_tracks(playlist_URI)
tracks = results['items']
while results['next']:
   results = sp.next(results)
   tracks.extend(results['items'])
track_artist = track_uris[99]
features  = []
for i in range(100):
    features.append(sp.audio_features(track_uris)[i])
details = []
emotions = ["happy","sad","fear","normal","angry"]
for i,track in enumerate(tracks):
  track_name = track['track']['name']
  track_artist = track['track']['artists'][0]['name']
  track_album = track['track']['album']['name']
  track_popularity = track['track']['popularity']
  track_emotion = random.choice(emotions)
  track_acousticness = features[i]['acousticness']
  track_id = features[i]['id']
  track_energy = features[i]['energy']
  track_loudness = features[i]['loudness']
  track_liveness = features[i]['liveness']
  track_danceability = features[i]['danceability']
  track_speechiness = features[i]['speechiness']
  track_instrumentalness = features[i]['instrumentalness']
  track_tempo = features[i]['tempo']


  details.append({
    'name': track_name,
    'artist': track_artist,
    'album': track_album,
    'popularity': track_popularity,
    'emotion': track_emotion,
    'acousticness': track_acousticness,
    'id': track_id,
    'energy': track_energy,
    'loudness': track_loudness,
    'liveness': track_liveness,
    'danceability': track_danceability,
    'speechiness': track_speechiness,
    'instrumentalness': track_instrumentalness,
    'tempo': track_tempo

  })  
df1 = pd.DataFrame(details)
df1['album'] = df1['album'].str.split('(').str[0]
c = df1
df1['popularity'] = df1['popularity'].apply(lambda x: int(x/5))
emotions_encoded = pd.get_dummies(df1['emotion'], prefix='emotion')
emotions_df = df1
d = ['name','artist','album','id']
df1 = df1[d]
emotions_df = emotions_df.drop(['name','artist','album','id','popularity','emotion'], axis=1)
emotions_normalized = (emotions_df - emotions_df.mean()) / emotions_df.std()  
emotions_normalized = emotions_df - emotions_df.mean()
train = pd.concat([emotions_encoded, emotions_normalized], axis=1)
normalized = pd.concat([df1, emotions_encoded, emotions_normalized], axis=1)
nn = NearestNeighbors(n_neighbors=20, algorithm='ball_tree')
nn.fit(train)
print(recommend_songs(finalev, 10))
sp.close()
