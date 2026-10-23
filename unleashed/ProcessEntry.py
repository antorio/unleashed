class ProcessEntry:
    def __init__(self, filename: str, start: int, end: int, fps: float):
        self.filename = filename
        self.finalname = None
        self.completed = False      # the output file exists and is final (not a partial __temp)
        self.destination = None     # a video's / GIF's final file, decided before rendering
        self.failure = ''           # why no file was made (shown by the Face Swap tab)
        self.no_audio = False       # saved without the source's sound (it could not be copied)
        self.gpu_recovered = 0      # GPU errors a retry fixed (onnx_guard)
        self.gpu_failed = []        # where a GPU error was not fixed ('frame 4647', a file name)
        # set only by the Face Swap tab for a file with a painted keep-original
        # mask (the editor layer); files without keep the run's own mask
        self.startframe = start
        self.endframe = end
        self.fps = fps