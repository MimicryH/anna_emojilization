import os.path
import xlrd
import numpy as np
import pandas as pd
import time
import socket
import struct

from tools.asr_xunfei import ASR_Xunfei
# from tools.tool_baidu import Tool_Baidu
from tools.SER_matlab import SERMatlab
from tools.TER_deepmoji import TERDeepMoji
from tools import inserting
import json
from data.visual_map import ScatterDebugger

WEIGHT_SPEECH = 5

SEMANTIC_EMOJI = [6, 13, 14, 17, 21, 33, 36, 38, 40, 42, 56]
EXLUSIVE_EMOJI = [11, 48]


class Emojilizer():
    def __init__(self):
        basedir = os.path.dirname(os.path.abspath(__file__))
        tabledir = os.path.join(basedir, 'data/emoji_coordinate2.000.xls')
        self.emoji_coordinate = self.readexcel(tabledir)
        self.ser = SERMatlab()
        self.ter = TERDeepMoji()
        self.asr = ASR_Xunfei()
        # self.tool_baidu = Tool_Baidu()
        self.speechEmoVec = None
        self.textEmojiList = []
        self.textEmoVec = []
        self.fuVecList = []
        self.rawText = None
        self.debugger = ScatterDebugger()

    def speechEmo(self, wavpath):
        return self.ser.speechEmo(wavpath)

    def deepmoji(self, text):
        return self.ter.textEmo(text)

    def readexcel(self, filepath):
        workbook = xlrd.open_workbook(filepath)
        # 抓取所有sheet页的名称
        worksheets = workbook.sheet_names()
        #print('worksheets is %s' % worksheets)
        # 定位到sheet1
        worksheet1 = workbook.sheet_by_name(u'Sheet1')
        num_rows = worksheet1.nrows
        #print('num_rows',num_rows)
        num_cols = 3
        #print('num_cols',num_cols)
        emoji_coordinate=np.zeros([64,2])
        for rown in range(1,num_rows):
            #print('rows:',rown)
            for coln in range(1,num_cols):
                #print('cols:',coln)
                cell = worksheet1.cell_value(rown, coln)
                emoji_coordinate[rown-1][coln-1]=cell
                #print(cell)
        #print(emoji_coordinate)
        return emoji_coordinate

    def fuse(self, emojilist, emotiontuple):
        if emotiontuple[0] in ["happy", "sad", "surprise", "angry"]:
            w1_audio = 10  # 语音通道的融合权值
        elif emotiontuple[0] == "neutral":
            w1_audio = 10 * np.random.rand()
        elif emotiontuple[0] in ["seemhappy", "seemsad", "seemtmotion", "seemangry"]:
            w1_audio = 4  # 语音通道的融合权值
        elif emotiontuple[0] == "emotion2":
            w1_audio = 5
        else:
            w1_audio = 1
        w1_text = 1  # 文本通道的融合权值
        w2_audio = WEIGHT_SPEECH  # 语音通道的扩展权值
        w1_audio = w1_audio * w2_audio
        w2_text = 1.8  # 文本通道权值
        # emotion = np.power(emotiontuple[1], 3)  # 先将语音通道算出的情感坐标先过x3后再线性扩展
        emotion = emotiontuple[1]
        vectorList = []
        for i in range(len(emojilist)):
            t_emoji = self.emoji_coordinate[emojilist[i]]  # emoji的二维坐标
            self.textEmoVec.append(t_emoji)
            # print("纯文本的情感向量坐标", t_emoji)
            self.debugger.plot_text(t_emoji[0], t_emoji[1])
            if emojilist[i] in SEMANTIC_EMOJI:
                vectorList.append(t_emoji)
            else:
                t_emoji = t_emoji * w2_text  # 将纯文本通道的情感坐标进行扩展
                vector = (t_emoji * w1_text + emotion * w1_audio) / (
                        w1_text + w1_audio)  # 将纯文本通道的坐标与语音通道的坐标做加权平均得到融合后的情感坐标：vector
                self.debugger.plot_fused(vector[0], vector[1])
                vectorList.append(vector)
        return vectorList

    def getEmotionVecList(self, wavepath):
        text = self.asr.getTextOf(wavepath)
        # text = self.tool_baidu.asr(wavepath)
        if text == "你说啥，我没听清。":
            return False
        else:
            self.textEmojiList, short_sentences = self.deepmoji(text)
            emotiontuple = self.speechEmo(wavepath)
            self.speechEmoVec = emotiontuple[1]
            vectorList = self.fuse(self.textEmojiList, emotiontuple)
            return vectorList, short_sentences

    def getEmojiList(self, wavepath, text):
        self.textEmoVec = []
        self.fuVecList = []
        if text == "":
            print("asr starts")
            text = self.asr.getTextOf(wavepath)
        # text = self.tool_baidu.asr(wavepath)
        self.rawText = text
        print("self.rawText = ", text)
        if text == "":
            return [], []
        else:
            emojilist, short_sentences = self.deepmoji(text)
            emotiontuple = self.speechEmo(wavepath)
            self.speechEmoVec = emotiontuple[1]
            vectorList = self.fuse(emojilist, emotiontuple)
            self.fuVecList = vectorList
            new_emoji = []
            for i in range(len(emojilist)):
                dist = np.sqrt(
                    np.sum(np.square(vectorList[i] - self.emoji_coordinate), axis=1, keepdims=True))  # 将融合后的情感坐标对64个emoji的坐标求欧式距离
                temp = np.where(dist == np.min(dist))  # 找出最近邻的emoji对应的坐标
                temp = int(temp[0][0])
                if not temp in EXLUSIVE_EMOJI:
                    new_emoji.append(temp)  # 将每一个短句对应的新的emoji写入new_emoji中
            return new_emoji, short_sentences

    def emojilize(self, wavepath, text=""):
        new_emoji, short_sentences = self.getEmojiList(wavepath, text)
        dict = {}
        length = len(new_emoji) if len(new_emoji) < len(short_sentences) else len(short_sentences)
        for i in range(length):
            print(i, short_sentences[i], new_emoji[i])
            dict[short_sentences[i]] = new_emoji[i]
        print(dict)
        f = inserting.insert_dic(dict)
        result = {"e_text": f, "text": short_sentences, "emoji_list": new_emoji, "vec_text": self.textEmoVec, "vec_speech": self.speechEmoVec.tolist(), "vec_fu_list": self.fuVecList,"raw_text": self.rawText}
        return result

def unit_test():
    start = time.clock()
    e = Emojilizer()
    elapsed = (time.clock() - start)
    print("Initialization Time used:", elapsed)
    start = time.clock()
    r = e.emojilize('E:\DoobiePJ\doobieBot\SerDual\static\\test_audio\\3.wav')
    print("结果为:", r)
    elapsed = (time.clock() - start)
    print("Onetime Emojilization Time used:", elapsed)
    start = time.clock()
    r = e.emojilize('E:\DoobiePJ\doobieBot\SerDual\static\\test_audio\\4.wav')
    print("结果为:", r)
    elapsed = (time.clock() - start)
    print("Onetime Emojilization Time used:", elapsed)
    start = time.clock()
    r = e.emojilize('E:\DoobiePJ\doobieBot\SerDual\static\\test_audio\\5.wav')
    print("结果为:", r)
    elapsed = (time.clock() - start)
    print("Onetime Emojilization Time used:", elapsed)
    start = time.clock()
    r = e.emojilize('E:\DoobiePJ\doobieBot\SerDual\static\\test_audio\\6.wav')
    print("结果为:", r)
    elapsed = (time.clock() - start)
    print("Onetime Emojilization Time used:", elapsed)

def emojilizationService():
    e = Emojilizer()
    print('Emojilizer is ready')
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = 'localhost'  # 获取本地主机名
    port = 12345  # 设置端口
    s.bind((host, port))  # 绑定端口
    s.listen(5)  # 等待客户端连接
    while True:
        print('Waiting for connection...')
        conn, addr = s.accept()     # 建立客户端连接
        with conn:
            print("socket 连接成功! ", addr)
            fileinfo_size = struct.calcsize('128sl')
            buf = conn.recv(fileinfo_size)
            if buf:
                filename, filesize = struct.unpack('128sl', buf)
                fn = filename.strip(b'\00')
                new_filename = os.path.join('./upload/', 'new_' + fn.decode())
                recvd_size = 0  # 定义已接收文件的大小
                fp = open(new_filename, 'wb')
                print('start receiving...')
            while not recvd_size == filesize:
                if filesize - recvd_size > 1024:
                    data = conn.recv(1024)
                    recvd_size += len(data)
                else:
                    data = conn.recv(filesize - recvd_size)
                    recvd_size = filesize
                fp.write(data)
            fp.close()
            print('end receiving...')
            r = e.emojilize(new_filename)
            r = json.dumps(r)
            r = str.encode(r)
            print(len(r), r)
            conn.send(struct.pack(b'l', len(r)))
            conn.sendall(r)
            time.sleep(2)
            conn.close()

def gen_huikua_emojis():
    e = Emojilizer()
    huikua_DF = pd.read_excel('E:\\DoobiePJ\\doobieBot\\huikua_mustwords(1).xlsx')
    emojilized_huikua_DF = pd.DataFrame(columns=['id', 'huikua', 'qiukua_id', 'mustword_list', 'kua_gender'])
    for index, row in huikua_DF.iterrows():
        # print(row)
        emojilist, short_sentences = e.deepmoji(row.values[1])
        dict = {}
        length = len(emojilist) if len(emojilist) < len(short_sentences) else len(short_sentences)
        for i in range(length):
            # print(i, short_sentences[i], emojilist[i])
            dict[short_sentences[i]] = emojilist[i]
        print(dict)
        f = inserting.insert_dic(dict)
        emojilized_huikua_DF.loc[index] = {'id': row.values[0], 'huikua': f, 'qiukua_id': row.values[2], 'mustword_list': row.values[3], 'kua_gender': row.values[4]}
    emojilized_huikua_DF.to_excel("emojilized_huikua.xlsx", index=False, encoding='utf8')


def add_huikua_emoji_list():
    import re
    list = [u'\U0001F602', u'\U0001F612', u'\U0001F629', u'\U0001F62D', u'\U0001F60D',u'\U0001F614', u'\U0001F44C',
        u'\U0001F60A', u'\u2764', u'\U0001F60F', u'\U0001F601', u'\U0001F3B6', u'\U0001F633', u'\U0001F4AF',
        u'\U0001F634', u'\U0001F60C', u'\u263A', u'\U0001F64C', u'\U0001F495', u'\U0001F611', u'\U0001F605',
        u'\U0001F64F', u'\U0001F615', u'\U0001F618', u'\u2665', u'\U0001F610', u'\U0001F481', u'\U0001F61E',
        u'\U0001F648', u'\U0001F62B', u'\u270C', u'\U0001F60E', u'\U0001F621', u'\U0001F44D', u'\U0001F625',
        u'\U0001F62A', u'\U0001F60B', u'\U0001F624', u'\U0001F91A', u'\U0001F637', u'\U0001F44F', u'\U0001F440',
        u'\U0001F52B', u'\U0001F623', u'\U0001F608', u'\U0001F613', u'\U0001F494', u'\U0001F49F', u'\U0001F3A7',
        u'\U0001F64A', u'\U0001F609', u'\U0001F480', u'\U0001F616', u'\U0001F604', u'\U0001F61C', u'\U0001F620',
        u'\U0001F645', u'\U0001F4AA', u'\U0001F44A', u'\U0001F49C', u'\U0001F496', u'\U0001F499', u'\U0001F62C',
        u'\u2728']
    pattern = r'\U0001F3B6|\U0001F3A7'
    emojilized_huikua_DF = pd.DataFrame(columns=['id', 'huikua', 'qiukua_id', 'mustword_list', 'kua_gender', 'emoji_list'])
    row_index = 0
    for i in range(1, 23):
        huikua_DF = pd.read_excel('E:\\DoobiePJ\\doobieBot\\kwa\\doobiebot\\emojilization\\emojilized_huikua%d.xlsx' % i)
        for index, row in huikua_DF.iterrows():
            print(row)
            emoji_list = []
            for i, element in enumerate(list):
                if element in row.values[1]:
                    emoji_list.append(str(i))
            huikua_revised = re.sub(pattern, u'\u2728', row.values[1])
            print(' '.join(emoji_list))
            emojilized_huikua_DF.loc[index + row_index] = {'id': row.values[0], 'huikua': huikua_revised,
                                               'qiukua_id': row.values[2], 'mustword_list': row.values[3],
                                               'kua_gender': row.values[4], 'emoji_list': ' '.join(emoji_list)}
        row_index += huikua_DF.shape[0]
    emojilized_huikua_DF.to_excel("emojilized_huikua.xlsx", index=False, encoding='utf8')

if __name__ == '__main__':
    # emojilizationService()
    e = Emojilizer()
    print(e.emojilize('to_do_list/0.wav'))
    e.debugger.show()
    # gen_huikua_emojis()
    # add_huikua_emoji_list()