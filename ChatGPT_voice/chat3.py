import openai
import speech_recognition as sr
import pyttsx3


# Initialize OpenAI API
openai.api_key = "YOUR_API_KEY_HERE"
# Initialize the text to speech engine
engine = pyttsx3.init()
engine.setProperty('voice', 'en-US')


def transcribe_audio_to_text(filename):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except:
        print("Unknown error occurred")


def generate_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=4000,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response["choices"][0]["text"]


def speak_text(text):
    engine.say(text)
    engine.runAndWait()


def main():
    while True:
        # Wait for user to say "yo"
        print("Say 'Yo' to start recording your question or 'Goodbye' to exit the program.")
        with sr.Microphone() as source:
            recognizer = sr.Recognizer()
            audio = recognizer.listen(source)
            try:
                transcription = recognizer.recognize_google(audio)
                if transcription.lower() == "yo":
                    # Record audio
                    filename = "input.wav"
                    print("Say your question")
                    with sr.Microphone() as source:
                        recognizer = sr.Recognizer()
                        source.pause_threshold = 1
                        audio = recognizer.listen(source, phrase_time_limit=None, timeout=None)
                        with open(filename, "wb") as f:
                            f.write(audio.get_wav_data())

                    # Transcript audio to text
                    text = transcribe_audio_to_text(filename)
                    if text:
                        print(f"You said: {text}")

                        # Generate the response
                        response = generate_response(text)
                        print(f"Chat GPT-3 says: {response}")

                        # Read response using GPT-3
                        speak_text(response)
                elif transcription.lower() == "goodbye":
                    # Terminate the program
                    print("Exiting the program. Goodbye!")
                    return
            except Exception as e:
                print("An error occurred: {}".format(e))


if __name__ == "__main__":
    main()
