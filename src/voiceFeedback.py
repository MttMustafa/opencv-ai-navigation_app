import pyttsx3

# reads messages in selected language

def speak(message):
    engine = pyttsx3.init()
    
    # sets the speed of the speech
    engine.setProperty("rate", 200)

    #selects the language from voices
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[64].id)
    
    # speaks the message
    engine.say(message)
    engine.runAndWait()