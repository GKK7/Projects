import openai
import speech_recognition as sr
import platform
import logging
import sys

# Initialize OpenAI API
openai.api_key = "YOUR_API_KEY_HERE"

# Initialize the text to speech engine
if platform.system() == "Windows":
    import win32com.client
    engine = win32com.client.Dispatch("SAPI.SpVoice")
else:
    import os
    import pyttsx3
    engine = pyttsx3.init()

# Set up logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def transcribe_audio_to_test(filename):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.RequestError as e:
        logging.error("Could not transcribe audio: {}".format(e))
    except sr.UnknownValueError as e:
        logging.error("Could not transcribe audio: {}".format(e))


def generate_response(prompt):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=4000,
            n=1,
            stop=None,
            temperature=0.5,
        )
        return response["choices"][0]["text"]
    except openai.Error as e:
        logging.error("Could not generate response: {}".format(e))


def speak_text(text):
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        logging.error("Could not speak text: {}".format(e))


def main():
    while True:
        # Wait for user to say "yo"
        logging.info("Say 'Yo' to start recording your question")
        with sr.Microphone() as source:
            recognizer = sr.Recognizer()
            audio = recognizer.listen(source)
            try:
                transcription = recognizer.recognize_google(audio)
                if transcription.lower() == "yo":
                    # Record audio
                    filename = "input.wav"
                    logging.info("Say your question")
                    with sr.Microphone() as source:
                        recognizer = sr.Recognizer()
                        source.pause_threshold = 1
                        audio = recognizer.listen(source, phrase_time_limit=None, timeout=None)
                        with open(filename, "wb") as f:
                            f.write(audio.get_wav_data())

                    # Transcribe audio to text
                    text = transcribe_audio_to_test(filename)
                    if text:
                        logging.info("You said: {}".format(text))

                        # Generate response
                        response = generate_response(text)
                        if response:
                            logging.info("Chat GPT-3 says: {}".format(response))

                            # Speak response
                            speak_text(response)
            except sr.RequestError as e:
                logging.error("Could not recognize speech: {}".format(e))
            except sr.UnknownValueError as e:
                logging.error("Could not recognize speech: {}".format(e))


if __name__ == "__main__":
    main()
