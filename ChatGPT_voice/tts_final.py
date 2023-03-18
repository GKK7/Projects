import openai
import speech_recognition as sr
import pyttsx3

# Set up OpenAI API key
openai.api_key = "YOUR_API_KEY_HERE"

# Set up text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('voice', 'en-US')

def transcribe_audio_to_text(filename):
    """
    Transcribes audio file to text using Google Speech Recognition.
    """
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except:
        return None

def generate_response(prompt):
    """
    Generates a response to a prompt using OpenAI's GPT-3 language model.
    """
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response["choices"][0]["text"].strip()

def speak_text(text):
    """
    Uses text-to-speech engine to speak a given text.
    """
    engine.say(text)
    engine.runAndWait()

def main():
    """
    Main function that listens for voice commands and responds using OpenAI's GPT-3 language model.
    """
    # Listen for "Hey Bot" wake word
    r = sr.Recognizer()
    while True:
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source)
        try:
            command = r.recognize_google(audio).lower()
            if "hey bot" in command:
                print("How can I help you?")
                speak_text("How can I help you?")
                break
        except:
            pass

    # Loop for listening and responding to user's queries
    while True:
        # Listen for user's query
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source)
        try:
            query = r.recognize_google(audio).lower()
            print("You said: " + query)

            # Stop the program if the user says "goodbye"
            if "goodbye" in query:
                print("Goodbye!")
                speak_text("Goodbye!")
                return

            # Generate and speak response to user's query
            response = generate_response(query)
            print("Bot says: " + response)
            speak_text(response)

        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))

if __name__ == "__main__":
    main()