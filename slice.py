# from pydub.silence import split_on_silence

from pydub import AudioSegment
import os

sound = AudioSegment.from_file("/Users/dave/Desktop/asr/zhaojing.MP3", format="mp3")

import pandas as pd

df = pd.read_excel("/Users/dave/Desktop/asr/text_zhaojin.xlsx")

df.head()

for index, row in df.iterrows():
    s = sound[row["bg"]:row["ed"]]
    s.export(f"./{index}.mp3", format="mp3")