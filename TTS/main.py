from gtts import gTTS
import playsound

text = "안녕하세요 저는 면접관 입니다"
tts = gTTS(text=text, lang="ko")
tts.save("response.mp3")

playsound.playsound("response.mp3")
