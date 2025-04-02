from kokoro import KModel, KPipeline
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import io
import soundfile as sf

# Initialize models and pipelines
CUDA_AVAILABLE = torch.cuda.is_available()
models = {gpu: KModel().to('cuda' if gpu else 'cpu').eval() for gpu in [False] + ([True] if CUDA_AVAILABLE else [])}
pipelines = {lang_code: KPipeline(lang_code=lang_code, model=False) for lang_code in 'abj'}
pipelines['a'].g2p.lexicon.golds['kokoro'] = 'kˈOkəɹO'
pipelines['b'].g2p.lexicon.golds['kokoro'] = 'kˈQkəɹQ'

# Available voices
VOICES = {
    'af_heart': 'af_heart',
    'af_bella': 'af_bella',
    'af_nicole': 'af_nicole',
    'af_aoede': 'af_aoede',
    'af_kore': 'af_kore',
    'af_sarah': 'af_sarah',
    'af_nova': 'af_nova',
    'af_sky': 'af_sky',
    'af_alloy': 'af_alloy',
    'af_jessica': 'af_jessica',
    'af_river': 'af_river',
    'am_michael': 'am_michael',
    'am_fenrir': 'am_fenrir',
    'am_puck': 'am_puck',
    'am_echo': 'am_echo',
    'am_eric': 'am_eric',
    'am_liam': 'am_liam',
    'am_onyx': 'am_onyx',
    'am_santa': 'am_santa',
    'am_adam': 'am_adam',
    'jf_alpha': 'jf_alpha',
    'jf_gongitsune': 'jf_gongitsune',
    'jf_nezumi': 'jf_nezumi',
    'jf_tebukuro': 'jf_tebukuro',
    'jm_kumo': 'jm_kumo',
}

# Load all voices
for v in VOICES.values():
    pipelines[v[0]].load_voice(v)

app = FastAPI(
    title="Text-to-Speech API",
    description="API for converting text to speech using various voices",
    version="1.0.0"
)

@app.get("/voices")
async def list_voices():
    """List all available voices"""
    return {"voices": list(VOICES.keys())}

@app.post("/tts")
async def text_to_speech(text: str, voice: str = "jf_alpha", speed: float = 1.0):
    """
    Convert text to speech using the specified voice
    
    Parameters:
    - text: The text to convert to speech
    - voice: The voice to use (default: jf_alpha)
    - speed: Speech speed (0.5 to 2.0, default: 1.0)
    """
    if voice not in VOICES:
        raise HTTPException(status_code=400, detail=f"Voice '{voice}' not found. Use /voices endpoint to see available voices.")
    
    if not 0.5 <= speed <= 2.0:
        raise HTTPException(status_code=400, detail="Speed must be between 0.5 and 2.0")
    
    try:
        # Generate audio using the pipeline
        pipeline = pipelines[voice[0]]
        pack = pipeline.load_voice(voice)
        
        # Get the first audio segment
        for _, ps, _ in pipeline(text, voice, speed):
            ref_s = pack[len(ps)-1]
            try:
                audio = models[False](ps, ref_s, speed)
                # Convert to numpy array
                audio_np = audio.numpy()
                
                # Create a BytesIO object to store the audio
                audio_buffer = io.BytesIO()
                
                # Save the audio as WAV file
                sf.write(audio_buffer, audio_np, 24000, format='WAV')
                audio_buffer.seek(0)
                
                # Return the audio stream
                return StreamingResponse(
                    audio_buffer,
                    media_type="audio/wav",
                    headers={
                        "Content-Disposition": f"attachment; filename=output.wav"
                    }
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        raise HTTPException(status_code=500, detail="No audio generated")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=40001)