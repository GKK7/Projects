import openai
import speech_recognition as sr
import pyttsx3

# Initialize OpenAI API. Insert your API key that you can get for free at platform.openai
openai.api_key = "YOUR_API_KEY_HERE"

# Initialize the text to speech engine with the python ttsx3 library
engine = pyttsx3.init()
engine.setProperty('voice', 'en-US')

# Function transcribes user audio to text and records it
def transcribe_audio_to_text(filename):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except:
        print("Unknown error occurred")

# Basic settings for generating the response
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

# Returning the response
def speak_text(text):
    engine.say(text)
    engine.runAndWait()

# Main function
def main():
    filename = "conversation_log.txt"
    mode = "a"  # Append mode
    with open(filename, mode=mode, encoding='utf-8') as file:
        while True:
            # Wait for user to say "hello"
            print("Say 'Hello' to start recording your question or 'Goodbye' to exit the program.")
            with sr.Microphone() as source:
                recognizer = sr.Recognizer()
                audio = recognizer.listen(source)
                try:
                    transcription = recognizer.recognize_google(audio)
                    if transcription.lower() == "hello":
                        # Record audio
                        filename = "input.wav"
                        print("Ask your question")
                        with sr.Microphone() as source:
                            recognizer = sr.Recognizer()
                            source.pause_threshold = 1
                            audio = recognizer.listen(source, phrase_time_limit=None, timeout=None)
                            with open(filename, "wb") as f:
                                f.write(audio.get_wav_data())

                        # Transcribe audio input to text
                        text = transcribe_audio_to_text(filename)
                        if text:
                            print(f"You said: {text}")

                            # Generate the response
                            response = generate_response(text)
                            print(f"Chat GPT-3 says: {response}")

                            # TTS reads response using GPT3
                            speak_text(response)
                            # Ending the transcription by saying goodbye
                    elif transcription.lower() == "goodbye":
                            # Terminate the program
                            print("Exiting the program. Goodbye!")
                            return
                except Exception as e:
                    print(f"An error occurred: {e}")

if __name__ == "__main__":
            main()