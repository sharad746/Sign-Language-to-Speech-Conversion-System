import pyttsx3

engine = pyttsx3.init()

# Set properties for the speech
engine.setProperty('rate', 150)  # Set the speaking rate (words per minute)
engine.setProperty('volume', 1)  # Set the volume (float between 0 and 1)

# Get input text from the user
text = input("Enter the text you want to convert to speech: ")

# Convert text to speech

engine.say(text)
engine.runAndWait()
