# -*- coding: utf-8 -*-
import requests
import time
import hashlib
import base64
import json

URL = "http://api.xfyun.cn/v1/service/v1/iat"
APPID = "5c6baae9"
API_KEY = "cfb4699f77a050e56b6dd983b35ce7b4"


def getHeader(aue, engineType):
    curTime = str(int(time.time()))
    # curTime = '1526542623'
    param = "{\"aue\":\"" + aue + "\"" + ",\"engine_type\":\"" + engineType + "\"}"
    # print("param:{}".format(param))
    paramBase64 = str(base64.b64encode(param.encode('utf-8')), 'utf-8')
    # print("x_param:{}".format(paramBase64))

    m2 = hashlib.md5()
    m2.update((API_KEY + curTime + paramBase64).encode('utf-8'))
    checkSum = m2.hexdigest()
    # print('checkSum:{}'.format(checkSum))
    header = {
        'X-CurTime': curTime,
        'X-Param': paramBase64,
        'X-Appid': APPID,
        'X-CheckSum': checkSum,
        'Content-Type': 'application/x-www-form-urlencoded; charset=utf-8',
    }
    # print(header)
    return header


def getBody(filepath):
    binfile = open(filepath, 'rb')
    data = {'audio': base64.b64encode(binfile.read())}
    # print(data)
    # print('data:{}'.format(type(data['audio'])))
    # print("type(data['audio']):{}".format(type(data['audio'])))
    return data


aue = "raw"
engineType = "sms16k"
class ASR_Xunfei():
    def getTextOf(self, audio_file):
        # try:
        r = requests.post(URL, headers=getHeader(aue, engineType), data=getBody(audio_file))
        r_str = r.content.decode('utf-8')
        print("asr_result: ", r_str)
        r_json = json.loads(r_str)
        if r_json['code'] == '10105' or r_json['data'] == '':
            print("asr returned blank string")
            return ""
        else:
            return r_json['data']
        # except:
        #     print("ASR_Xunfei.getTextOf() error!")
        #     return ""

if __name__ == '__main__':
    asr = ASR_Xunfei()
    print(asr.getTextOf('E:\DoobiePJ\doobieBot\SerDual\static\\test_audio\\1.wav'))