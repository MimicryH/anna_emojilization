import jieba
import jieba.analyse
import re


'''
输入参数为一个字典，键为文本，值为emoji的序号。
'''


def insert_dic(dic):
    f = ''
    for i in dic.keys():
        # print("i=", i)
        # print("dic[i] = ", dic[i])
        t = insert(i, dic[i])
        f = f + t
    return f


"""
将emoji插入到句子中,默认为模式1
模式1为按照关键词插入，找到句子中权重最大的关键词并将emoji插入到关键词的后面。
模式2为将emoji替换句子中的标点符号。
模式3为关键词与标点都加表情。
模式4为只在最后加表情。
模式5为根据标点符号插入表情，'，'和'。'把表情插在前边，其他标点符号则插在后边。
"""


def insert(text, emoji_num, mode=5):
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
    pattern = r',|\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|。|、|；|‘|’|【|】|·|！| |…|（|）|\？'
    # ban the musical notes and the headphone
    pattern2 = r'\U0001F3B6|\U0001F3A7'
    emoji = list[emoji_num]
    emoji = re.sub(pattern, u'\u2728', emoji)
    if mode == 1:
        t = jieba.analyse.extract_tags(text, topK=20, withWeight=False, allowPOS=())
        # print(t)
        for key in t:
            print(key)
            text = text.replace(key, key + emoji)
        return text
    elif mode == 2:
        f = re.sub(pattern, emoji, text)
        return f
    elif mode == 3:
        t = jieba.analyse.extract_tags(text, topK=20, withWeight=False, allowPOS=())
        for key in t:
            print(key)
            text = text.replace(key, key + emoji)
        f = re.sub(pattern, emoji, text)
        return f
    elif mode == 4:
        f = text + emoji
        return f
    elif mode == 5:
        f = ''
        l = 1
        while l:
            l = re.search(pattern, text)
            if l is None:
                if f == '':
                    f = text + emoji
                break
            s = l.end(0)
            if l.group(0)==',' or l.group(0)=='.'or l.group(0)=='，'or l.group(0)=='。':
                #if(s > 4):
                f = f + text[:s-1]+emoji + text[s-1:s]
                #else:
                #    f = f + text[:s]
            else:
                #if(s > 4):
                f = f + text[:s] + emoji
                #else:
                #    f = f + text[:s]
            text = text[s:]
        return f


if __name__ == '__main__':
    dict = {'wosssssss': 35}
    print(insert_dic(dict))