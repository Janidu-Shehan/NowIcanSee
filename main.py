import os
import pyaudio
import wave
from google.cloud import speech
from google import genai
import time
import cv2
from google.genai import types
from PIL import Image

# Load the environment variables from the .env file
load_dotenv()

# Get the API keys and credentials from the environment variables
client_answer = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

# Set the Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

# Initialize Google Cloud Speech client
client = speech.SpeechClient()
# System instruction
system_instruction_video = """
You are an advanced AI model designed to generate descriptive narrations for blind or visually impaired users based on video analysis. 

When generating descriptions, follow these guidelines:

1. **Clarity & Simplicity**:
   - Use simple, clear, and easy-to-understand language.
   - Avoid technical terms or unnecessary complexity.

2. **Contextual Awareness**:
   - Identify key objects, actions, and settings in the scene.
   - Provide spatial awareness (e.g., "A person is walking from the left side of the street toward a bus stop on the right").
   - Include important environmental cues (e.g., "It's a rainy day, and people are carrying umbrellas").

3. **Structured and Concise Output**:
   - Start with a **general scene overview** before describing details.
   - Use **short, natural-sounding sentences**.
   - Prioritize moving objects and interactions.

4. **Real-time Considerations**:
   - If the scene is **dynamic**, provide sequential updates (e.g., "A cyclist is approaching from the left and passing a pedestrian on the right").
   - Use **smooth transitions** between updates (e.g., "Now, the cyclist has moved past the pedestrian and is heading toward the intersection").

5. **Tone & Engagement**:
   - Maintain a **neutral yet engaging** tone.
   - Do not infer emotions unless explicitly visible (e.g., "The child is smiling and waving").
"""

system_instruction_image = """
You are an advanced AI model designed to generate **descriptive narrations** for blind or visually impaired users based on image analysis.

When generating descriptions, follow these structured guidelines:

1. **Clarity & Simplicity**:
   - Use **simple, clear, and direct language**.
   - Avoid complex or highly technical terms unless necessary.

2. **Contextual Awareness**:
   - Identify **key elements** in the image (e.g., people, objects, animals, landscapes).
   - Specify relationships between objects (e.g., "A woman is sitting at a table with a laptop and a cup of coffee").
   - Highlight environmental details (e.g., "The sky is bright blue with a few scattered clouds").

3. **Structured and Concise Output**:
   - **Start with a general overview**, then add key details.
   - Use **short, natural-sounding sentences**.
   - Prioritize **relevant elements** over minor or redundant details.

4. **Describing People & Objects**:
   - If there are **people**, describe their actions and general appearance (e.g., "A man wearing a red jacket is reading a book").
   - Mention **object placement and significance** (e.g., "A bicycle is leaning against a tree in the park").
   - If text is present in the image, **extract and include it** (e.g., "A sign in the background says 'Caution: Wet Floor'").

5. **Accessibility Considerations**:
   - Use **descriptive language** instead of assumptions (e.g., say *"A child is smiling"* instead of *"A happy child"*).
   - Avoid **guessing emotions or making subjective judgments** unless clearly visible.

6. **Tone & Engagement**:
   - Maintain a **neutral yet engaging tone**.
   - Ensure the description is **informative, but not overwhelming**.
   - Provide details that help **visualize the scene naturally**.

"""

# Function to record audio from the microphone
def record_audio(filename="audio.wav"):
    # Set up audio stream
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=1024)

    frames = []
    print("Recording... Speak now!")
    for _ in range(0, int(16000 / 1024 * 5)):  # Record for 5 seconds
        data = stream.read(1024)
        frames.append(data)

    print("Finished recording.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded audio to a WAV file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))

# Function to convert audio to text using Google Cloud Speech-to-Text
def speech_to_text(audio_file):
    with open(audio_file, 'rb') as audio:
        audio_content = audio.read()

    audio = speech.RecognitionAudio(content=audio_content)

    # Configuration for the speech recognition
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
    )

    # Recognize speech using the client
    response = client.recognize(config=config, audio=audio)

    # Display the results
    for result in response.results:
        print(result.alternatives[0].transcript)
        return (result.alternatives[0].transcript)



def capture_image():
    # Open the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return None

    print("Capturing image automatically...")

    # Capture the first frame (auto capture)
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image.")
        cap.release()
        return None

    # Convert the captured frame (OpenCV format) to PIL Image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return image

def text_to_caategory(question):
    system_instruction="You are an intelligent assistant capable of answering questions in multiple formats:1. If the question can be answered by text (e.g., general questions, facts, recipes), respond 'text'.2. If the question requires an image (e.g., questions about what is in front of the user or describing objects), respond 'image'.3. If the question requires video (e.g., questions about dynamic environments or what is happening around the user), respond 'video'.For each user query, analyze the content and determine whether the answer should be in text, image, or video format."
    response = client_answer.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=system_instruction),
        contents=[question])
    return(response.text)

def text_to_answer(question):
    response = client_answer.models.generate_content(
        model="gemini-2.0-flash",
        contents=[question])
    return(response.text)

def capture_video(duration: int = 4, video_filename: str = "captured_video.mp4"):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return None

    # Set video codec and create VideoWriter object to save the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For .mp4 files
    out = cv2.VideoWriter(video_filename, fourcc, 20.0, (640, 480))

    print(f"Recording video for {duration} seconds...")

    # Capture the video for the specified duration
    start_record_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Write the frame to the video file
        out.write(frame)

        # Display the frame while recording
        cv2.imshow("Recording Video", frame)

        # Stop recording after the specified duration
        if time.time() - start_record_time > duration:
            break

    # Release the camera and close the window
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Video saved as {video_filename}")

    return video_filename


# Main function to execute speech-to-text conversion
def main():
    record_audio()  # Record audio from microphone
    question=speech_to_text("audio.wav")  # Convert the recorded audio to text
    typ=text_to_caategory(question)
    typ=typ.replace("\n", "")
    typ.strip().lower()
    print(list(typ))
    print(type(typ))
    print(typ)
    if typ == 'text':
        answer=text_to_answer(question)
        print(answer)
    elif typ == 'image':
        image = capture_image()
        response = client_answer.models.generate_content(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(
            system_instruction=system_instruction_image),
            contents=[question, image]
        )
        answer=(response.text)
        print(answer)
    elif typ == 'video':
        video_filename = capture_video(duration=4)
        video_file = client_answer.files.upload(file=video_filename)
        while video_file.state.name == "PROCESSING":
            print('.', end='')
            time.sleep(1)
            video_file = client_answer.files.get(name=video_file.name)
        if video_file.state.name == "FAILED":
            raise ValueError(video_file.state.name)

        print('Video processing completed.')
        # Generate content using the video file
        response = client_answer.models.generate_content(
            model="gemini-1.5-pro",
            config=types.GenerateContentConfig(
                system_instruction=system_instruction_video),
            contents=[video_file, question]
        )
        # Print and return the response from the model
        print(response.text)

        # Delete the uploaded video file to clean up
        client_answer.files.delete(name=video_file.name)
    else:
        print("can u repeat the question")





if __name__ == '__main__':
    main()
