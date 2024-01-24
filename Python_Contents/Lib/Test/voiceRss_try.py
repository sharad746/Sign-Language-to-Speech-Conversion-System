import requests
import os

API_KEY = "76109602f2ff4641b0f9c637a7e7511c"

text = "Hello World"
url = "http://api.voicerss.org/"
params = {"key": API_KEY, "src": text}
response = requests.get(url,params)

with open("output.mp3", "wb") as f:
    f.write(response.content)

audio = os.system("start output.mp3")