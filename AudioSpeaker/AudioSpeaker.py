import io
from edge_tts import Communicate
from pydub import AudioSegment
import simpleaudio as sa

class AudioSpeaker:
    def __init__(self):
        pass
    
    
    async def speak(self, text: str) -> float:
        communicate = Communicate(text, voice="ko-KR-SunHiNeural")

        audio_buffer = bytearray()
        
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.extend(chunk["data"])
        
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_buffer), format="mp3")

        duration_sec = audio_segment.duration_seconds
        
        playback = sa.play_buffer(
            audio_segment.raw_data,
            num_channels = audio_segment.channels,
            bytes_per_sample = audio_segment.sample_width,
            sample_rate = audio_segment.frame_rate
        )

        playback.wait_done()
        return duration_sec

        