import matlab.engine
import numpy as np
import os.path

class SERMatlab():
    def __init__(self):
        self.engine = matlab.engine.start_matlab()
        matlabWorkPath = os.path.abspath(os.path.dirname(__file__)) + "/matlab/"
        print(matlabWorkPath)
        self.engine.addpath(matlabWorkPath)
        self.engine.savepath()
        self.engine.vl_compilenn
        self.engine.paramsinit()
        #设置情感基向量
        self.neutral=np.array([0,0])
        self.happy=np.array([1,1])
        self.sad=np.array([-1,-1])
        self.surprise=np.array([0,1])
        self.angry=np.array([-1,1])
        #预设情况判定
        self.preset=np.array([1,1,1,1,1])
        self.preset0=np.array([0,1,1,1,1])
        self.preset1=np.array([1,0,0,0,0])
        self.preset2=np.array([0,1,0,0,0])
        self.preset3=np.array([0,0,1,0,0])
        self.preset4=np.array([0,0,0,1,0])
        self.preset5=np.array([0,0,0,0,1])
        self.preset6=np.array([0,1,1,1,0])
        self.preset7=np.array([1,0,0,1,0])
        self.preset8=np.array([0,0,0,1,1])
        self.preset9=np.array([0,1,1,0,1])
        self.preset10=np.array([1,1,1,0,1])

    def speechEmo(self, wavPath):
        #引入语音模型结果
        label=self.engine.experi(wavPath)
        if isinstance(label, float):
            label = [[0, 0, 0, 0, 0]]
        lb=np.array(label)
        # print(lb)
        # print(lb==preset)
        # 进入决策层
        decide_array = lb == self.preset
        if np.linalg.norm(lb - self.preset) == 0. or np.linalg.norm(lb - self.preset) == 1.:
            # print("语音情感为neutral",self. neutral)
            return ("neutral", self.neutral)
        if np.linalg.norm(lb - self.preset1) == 0.:
            # print("语音情感为neutral", self.neutral)
            return ("neutral", self.neutral)
        elif np.linalg.norm(lb - self.preset2) == 0.:
            # print("语音情感为happy", self.happy)
            return ("happy", self.happy)
        elif np.linalg.norm(lb - self.preset3) == 0.:
            # print("语音情感为surprise", self.surprise)
            return ("suprise", self.surprise)
        elif np.linalg.norm(lb - self.preset4) == 0.:
            # print("语音情感为sad", self.sad)
            return ("sad", self.sad)
        elif np.linalg.norm(lb - self.preset5) == 0.:
            # print("语音情感为angry", self.angry)
            return ("angry", self.angry)
        elif np.linalg.norm(lb - self.preset6) == 0.:
            # print("语音情感为seemhappy", self.happy)
            return ("seemhappy", self.happy)
        elif np.linalg.norm(lb - self.preset7) == 0.:
            emotion = 0.5 * self.neutral + 0.5 * self.sad
            # print("语音情感为neutral+sad", emotion)
            return ("seememotion", emotion)
        elif np.linalg.norm(lb - self.preset8) == 0.:
            # print("语音情感为seemsad", self.sad)
            return ("seemsad", self.sad)
        # elif np.linalg.norm(lb-preset9)==0.:
        #    print("语音情感为seemangry", angry)
        #    return ("seemangry",angry)
        # elif np.linalg.norm(lb-preset10)==0.:
        #    print("语音情感为angry", angry)
        #    return angry
        else:
            # print(neutral)
            # print(lb[0])
            emotion = 0.83 * self.neutral * lb[0][0] + 0.672 * self.happy * lb[0][1] + 0.426 * self.surprise * lb[0][2] + 0.215 * self.sad * \
                      lb[0][3] + 0.239 * self.angry * lb[0][4]
            # print("语音情感为emotion", emotion)
            if lb.sum() == 3.:
                return ("emotion3", emotion) #3的意思就是出现3种情感例如[0,1,1,1,0]
            else:
                return ("emotion2", emotion) #2的意思是在那个matlab返回的5种情感中出现2种例如，[0,1,1,0,0]

if __name__ == '__main__':
    s = SERMatlab()
    s.speechEmo('E:\DoobiePJ\doobieBot\SerDual\static\\test_audio\\1.wav')

