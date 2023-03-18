import openai
import speech_recognition as sr
import pyttsx3
import pyaudio

# Initialize OpenAI API. Use my API key or your own
openai.api_key = "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"
# Initialize the text to speech engine
engine = pyttsx3.init()


def transcribe_audio_to_text(filename):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        print("Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from the Speech Recognition service; {e}")


def generate_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=4000,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response.choices[0].text


def speak_text(text):
    engine.say(text)
    engine.runAndWait()


def main():
    while True:
        # Wait for user to say "deepdream"
        print("Say 'deepdream' to start recording your question")
        with sr.Microphone() as source:
            recognizer = sr.Recognizer()
            audio = recognizer.listen(source)
            try:
                transcription = recognizer.recognize_google(audio)
                if transcription.lower() == "deepdream":
                    # Record audio
                    filename = "input.wav"
                    print("Say your question")
                    with sr.Microphone() as source:
                        recognizer = sr.Recognizer()
                        source.pause_threshold = 1
                        audio = recognizer.listen(source, phrase_time_limit=None, timeout=None)
                        with open(filename, "wb") as f:
                            f.write(audio.get_wav_data())

                    # Transcribe audio to text
                    text = transcribe_audio_to_text(filename)
                    if text:
                        print(f"You said: {text}")

                        # Generate the response
                        response = generate_response(text)
                        print(f"Chat GPT3 says: {response}")

                        # Read response using tts
                        speak_text(response)
            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand audio")
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")
            except Exception as e:
                print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()

