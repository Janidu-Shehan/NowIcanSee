# 🦾 Robot User Interaction Platform for Blind Persons

This project is a **robot-user interaction platform** designed to assist **visually impaired individuals**. It utilizes the **Gemini API** for advanced conversational AI capabilities, combined with **Google Cloud's Text-to-Speech** and **Speech-to-Text APIs** to create a seamless **voice-based interaction system**.

---

## 🌟 Features

- 🤖 **Conversational AI** powered by Gemini API.
- 🔊 **Text-to-Speech (TTS)** to convert responses into natural-sounding speech.
- 🎙️ **Speech-to-Text (STT)** to capture and process user commands.
- 🧑‍🦯 Designed specifically for **blind and visually impaired users**.
- ⚙️ Flexible architecture for integrating with various robotic hardware (if needed).

---

## 🛠️ Technologies Used

| Technology      | Purpose |
|------------------|---------|
| Python           | Main programming language |
| Gemini API       | Natural language processing (NLP) |
| Google Cloud TTS | Convert text responses into speech |
| Google Cloud STT | Convert speech input into text |

---

## 📦 Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/Janidu-Shehan/NowIcansee.git
    cd NowIcansee
    ```

2. Create and activate a virtual environment (recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate   # On Windows use: venv\Scripts\activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Set up Google Cloud credentials for Speech-to-Text and Text-to-Speech:
    - Create a **Google Cloud project**.
    - Enable **Cloud Speech-to-Text** and **Cloud Text-to-Speech** APIs.
    - Download the **service account key JSON file**.
    - Set the environment variable:
        ```bash
        export GOOGLE_APPLICATION_CREDENTIALS="path/to/your-service-account-file.json"
        ```

---

## 🚀 Usage

Run the main Python script:
```bash
python main.py
