from LLMInterviewer.LLMInterviewer import LLMInterviewer

class Main():
    def __init__(self):
        pass

    @staticmethod 
    def start():
        LLMInterviewer().run()

### main함수 호출시 사용
if __name__ == "__main__":
   Main.start() 