
import io
import torch
from pydub import AudioSegment

def chunkyify(input_file, target_length_sec=6.0):
    #chunkify handles multiple issues. It decodes and resamples, pushes audio to VAD to detect when sentences start and end,
    #then splits the audio into chunks which get passed to parakeet for batch transcription, returning text, and timestamps.
    
    #Resample to 16kHz, mono, 16-bit PCM WAV
    og_audio = AudioSegment.from_file(input_file)
    conversion16k = og_audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)

    wav_io = io.BytesIO()
    conversion16k.export(wav_io, format="wav")
    wav_io.seek(0)

    #VAD
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True)
    get_speech_timestamps, _, read_audio, _, _ = utils

    wav_data = read_audio(wav_io)
    speech_timestamps = get_speech_timestamps(wav_data, 
                                              model,
                                              sampling_rate=16000,
                                              min_silence_duration_ms=450, 
                                              min_speech_duration_ms=250)
    chunks_for_parakeet = []
    current_chunk_start = None
    
    for i, ts in enumerate(speech_timestamps):
        start_ms = int((ts['start'] / 16000) * 1000)
        end_ms = int((ts['end'] / 16000) * 1000)
        
        if current_chunk_start is None:
            current_chunk_start = start_ms
        
        if (end_ms - current_chunk_start) / 1000.0 >= target_length_sec or i == len(speech_timestamps) - 1:
            chunks_for_parakeet.append({
                "start_ms": current_chunk_start,
                "end_ms": end_ms,
                # Slice directly from high-quality original audio
                "audio_segment": og_audio[current_chunk_start:end_ms] 
            })
            current_chunk_start = None   
        
        print(f"Split podcast into {len(chunks_for_parakeet)} blocks.")

        for chunk in chunks_for_parakeet:
        