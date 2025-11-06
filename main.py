from fastapi import FastAPI, HTTPException, Response, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import base64
import json
import os
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define the input model for the text request
class TextInput(BaseModel):
    text: str
    output_format: str = "base64"  # Default output is base64, but can be binary
    text_speaker: str = "id_female_icha"  # Default voice

# Define the input model for the OpenAI TTS request
class OpenAITTSInput(BaseModel):
    input: str
    prompt: str = ""
    voice: str = "sage"
    vibe: str = "null"

# Split the text into chunks based on max length
def split_text_into_chunks(text, max_length=280):
    chunks = []
    while text:
        chunk = text[:max_length]
        text = text[max_length:]
        chunks.append(chunk)
    return chunks

# Generate audio for each chunk
def generate_audio(chunk, text_speaker="id_female_icha"):
    url = "https://api.tiktokv.com/media/api/text/speech/invoke/"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
        "User-Agent": "com.zhiliaoapp.musically/2022600030 (Linux; U; Android 7.1.2; es_ES; SM-G988N; Build/NRD90M;tt-ok/3.12.13.1)",
        "Cookie": "sessionid=581a1225c93f9b4bb9aacc49e4ebc5a9"
    }
    data = {
        "req_text": chunk,
        "speaker_map_type": 0,
        "aid": 1233,
        "text_speaker": text_speaker
    }
    response = requests.post(url, headers=headers, data=data)
    response_json = response.json()
    if response_json["status_code"] == 0:
        return response_json["data"]["v_str"]
    else:
        raise HTTPException(status_code=500, detail="Failed to generate audio")

# Concatenate base64 MP3 chunks
def concatenate_base64_mp3(encoded_files):
    final_audio_data = b""
    for i, encoded_data in enumerate(encoded_files):
        decoded_data = base64.b64decode(encoded_data)
        if i == 0:
            final_audio_data += decoded_data
        else:
            # If it's not the first chunk, we may need to remove ID3 tags
            header = decoded_data[:10]
            if header[:3] == b'ID3':  # Check if there is an ID3 tag
                tag_size = int.from_bytes(header[6:10], byteorder='big')
                decoded_data = decoded_data[10 + tag_size:]
            final_audio_data += decoded_data
    return final_audio_data

# Main endpoint to receive text and return TTS in base64 or binary
@app.post("/tts")
async def tts_endpoint(input: TextInput, response: Response):
    text = input.text
    output_format = input.output_format.lower()
    text_speaker = input.text_speaker
    
    # Split the input text into chunks if it's too long
    chunks = split_text_into_chunks(text)
    
    base64_strings = []
    for chunk in chunks:
        base64_string = generate_audio(chunk, text_speaker)
        base64_strings.append(base64_string)
    
    # Concatenate the base64 audio strings
    concatenated_audio = concatenate_base64_mp3(base64_strings)

    if output_format == "binary":
        # Return binary audio file
        response.headers["Content-Type"] = "audio/mpeg"
        return Response(content=concatenated_audio, media_type="audio/mpeg")
    elif output_format == "base64":
        # Convert binary audio to base64 and return as JSON
        final_base64_audio = base64.b64encode(concatenated_audio).decode("utf-8")
        return {"audio_base64": final_base64_audio}
    else:
        # Invalid format
        raise HTTPException(status_code=400, detail="Invalid output format. Choose 'base64' or 'binary'.")

# Generate audio for each chunk using OpenAI.FM
def generate_audio_openai(chunk, prompt, voice, vibe):
    url = "https://www.openai.fm/api/generate"
    
    headers = {
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
    }
    
    files = {
        "input": (None, chunk),
        "prompt": (None, prompt),
        "voice": (None, voice),
        "vibe": (None, vibe)
    }
    
    response = requests.post(url, headers=headers, files=files)
    
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Failed to generate audio from OpenAI.FM")
    
    return response.content

# Concatenate WAV audio files
def concatenate_wav_files(audio_chunks):
    if not audio_chunks:
        return b""
    
    # WAV file structure: 
    # - First 44 bytes is typically the header (can vary, but 44 is standard for PCM)
    # - Rest is audio data
    header_size = 44
    
    # Extract header from first chunk
    result = audio_chunks[0]
    
    # For subsequent chunks, skip the header and only add the data portion
    for chunk in audio_chunks[1:]:
        if len(chunk) > header_size:
            result += chunk[header_size:]
    
    # Update the data size in the header (bytes 4-7 = RIFF chunk size, bytes 40-43 = data chunk size)
    data_size = len(result) - header_size
    total_size = data_size + 36  # RIFF chunk size = data size + 36 for standard PCM WAV
    
    # Update RIFF chunk size (total file size - 8 bytes)
    result = result[:4] + total_size.to_bytes(4, byteorder='little') + result[8:]
    
    # Update data chunk size
    result = result[:40] + data_size.to_bytes(4, byteorder='little') + result[44:]
    
    return result

# New endpoint for OpenAI.FM TTS
@app.post("/tts-openai")
async def tts_openai_endpoint(input_data: OpenAITTSInput):
    # Split the input text into chunks of max 999 characters
    chunks = split_text_into_chunks(input_data.input, max_length=999)
    
    # Generate audio for each chunk
    audio_chunks = []
    for chunk in chunks:
        audio_data = generate_audio_openai(chunk, input_data.prompt, input_data.voice, input_data.vibe)
        audio_chunks.append(audio_data)
    
    # Concatenate audio chunks properly for WAV format
    final_audio = concatenate_wav_files(audio_chunks)
    
    # Convert binary audio to base64
    audio_base64 = base64.b64encode(final_audio).decode("utf-8")
    
    # Return as JSON with audio_base64 field
    return {"audio_base64": audio_base64}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

