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
print(recommend_songs('happy', 10))
