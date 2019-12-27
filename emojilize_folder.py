from emojilizer import Emojilizer
import pandas as pd
import numpy as np
import os
import time

basedir = os.path.dirname(os.path.abspath(__file__))
tabledir = os.path.join(basedir, 'xlsx/zhaojing1.xlsx')
df = pd.read_excel(tabledir, "Sheet1")
text_list = df["文本"]
emo_text_list = []
text_v = []
text_a = []
speech_v = []
speech_a = []
fusion_v = []
fusion_a = []

emojilizer = Emojilizer()

for index, text in enumerate(text_list):
    # if index>100:
    #     break
    time.sleep(1)
    print("to_do_list/" + str(index) + ".wav", text)
    emojilization = emojilizer.emojilize("to_do_list/" + str(index) + ".wav", text)
    print(emojilization)
    emojilizer.debugger.show()
    emo_text_list.append(emojilization['e_text'])
    list_v = []
    list_a = []
    for v in emojilization['vec_text']:
        list_v.append(v[0])
        list_a.append(v[1])
    text_v.append(np.mean(list_v))
    text_a.append(np.mean(list_a))
    speech_v.append(emojilization['vec_speech'][0])
    speech_a.append(emojilization['vec_speech'][1])
    list_v = []
    list_a = []
    for v in emojilization['vec_fu_list']:
        list_v.append(v[0])
        list_a.append(v[1])
    fusion_v.append(np.mean(list_v))
    fusion_a.append(np.mean(list_a))

df = pd.DataFrame({'emo_text':emo_text_list, 'text_v':text_v, 'text_a':text_a, 'speech_v':speech_v,
                   'speech_a':speech_a, 'fusion_v':fusion_v, 'fusion_a':fusion_a})

# df['emo_text_list'] = emo_text_list
# df['text_v'] = text_v
# df['text_a'] = text_a
# df['speech_v'] = speech_v
# df['speech_a'] = speech_a
# df['fusion_v'] = fusion_v
# df['fusion_a'] = fusion_a

df.to_csv("result1.csv")






