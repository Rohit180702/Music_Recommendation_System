import speech_recognition as sr

# Initialize recognizer
r = sr.Recognizer()

# Use microphone as source
with sr.Microphone() as source:

    # Set energy threshold to ambient noise level
    r.energy_threshold = 300

    # Dynamic energy threshold adjustment
    r.dynamic_energy_threshold = True
    r.dynamic_energy_adjustment_ratio = 1.5

    # Set minimum pause length
    r.pause_threshold = 0.5

    # Non-speaking duration
    r.non_speaking_duration = 0.3

    # Max time for listening
    r.operation_timeout = None

    # Phrase threshold
    r.phrase_threshold = 0.3

    # Max time for a phrase
    r.max_silence = 1

    # Calibrate the energy threshold
    r.adjust_for_ambient_noise(source, duration=0.5)

    # Print statement to prompt user to speak
    print("Speak...")

    # Listen to microphone and convert speech to text
    while 1:
        try:
            # Capture the audio
            audio = r.listen(source, phrase_time_limit=5)

            # Convert speech to text
            text = r.recognize_google(audio,language=  "EN_US")

            # Print the recognized text
            print("You said: " + text)
        
        except KeyboardInterrupt:
            print("\nKeyboard interrupt detected. Exiting...")
            break

        except sr.UnknownValueError:
            # If the speech cannot be recognized
            print("Could not understand audio")

        except sr.RequestError as e:
            # If there is an error from the API
            print("Error: {0}".format(e))
  
