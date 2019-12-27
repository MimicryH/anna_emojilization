import json
import re
import os.path
from deepmoji.sentence_tokenizer import SentenceTokenizer
from deepmoji.model_def import deepmoji_emojis
from tools.translate import google_translate
# import translate
import numpy as np

class TERDeepMoji():

    def __init__(self):
        self.load_deepmoji()

    def load_deepmoji(self):
        maxlen = 30
        batch_size = 32

        # print(PRETRAINED_PATH,VOCAB_PATH)
        # 获取模型参数
        self.root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.modelvocabpath = os.path.join(self.root,'deepmoji', 'model', 'vocabulary.json')
        self.modelprepath = os.path.join(self.root,'deepmoji','model', 'deepmoji_weights.hdf5')
        # print(self.modelvocabpath)
        print('Tokenizing using dictionary from {}'.format(self.modelvocabpath))
        with open( self.modelvocabpath, 'r') as f:
            vocabulary = json.load(f)
        self.st = SentenceTokenizer(vocabulary, maxlen)
        print('Loading model from {}.'.format(self.modelprepath))
        global model
        model = deepmoji_emojis(maxlen, self.modelprepath)
        model.summary()

    # 1 for Chinese  2 for English
    def split(self, text, mode='en'):
        pattern = r',|\.|/|;|`|\[|\]|<|>|\？|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|。|、|；|‘|’|【|】|·|！|…|（|）'
        if mode == 'ch':
            pattern = '('+pattern+')'
        l = re.split(pattern, text)
        result_list = []
        for phrase in l:
            if len(phrase) > 0 and phrase != ' ':
                result_list.append(phrase)
        if mode == 'ch':
            result_list_chinese = []
            for s in result_list:
                if re.search(pattern, s) == None:
                    result_list_chinese.append(s)
                elif len(result_list_chinese) > 0:
                    result_list_chinese[-1] = result_list_chinese[-1] + s
            return result_list_chinese
        else:
            return result_list

    def top_elements(self, array, k):
        ind = np.argpartition(array, -k)[-k:]  # ind 为array 中最大的k个数在array中的索引
        return ind[np.argsort(array[ind])][
               ::-1]  ##array[ind]得到array数组中排名前k 的数的值，这k个数不一定是降序排列的，所以np.argsort输出升序排列后的array(ind)数组的索引值,再ind后再倒序成为降序的k个值。

    def textEmo(self, text):
        pattern = r'\(|\)|（|）'
        text = re.sub(pattern, '，', text)
        tra = google_translate(text)
        print(text)
        print("tra")
        print(tra)
        TEST_SENTENCES = self.split(tra, 'en')
        TEST_SENTENCES_CHI = self.split(text, 'ch')
        print("TEST_SENTENCES: ")
        print(TEST_SENTENCES)

        tokenized, _, _ = self.st.tokenize_sentences(TEST_SENTENCES)
        prob = model.predict(tokenized)  # 模型完成对emoji的预测
        scores = []
        emosequence = []
        for i, t in enumerate(TEST_SENTENCES):
            t_tokens = tokenized[i]
            t_score = [t]
            t_prob = prob[i]
            ind_top = self.top_elements(t_prob, 5)
            t_score.append(sum(t_prob[ind_top]))
            t_score.extend(ind_top)
            t_score.extend([t_prob[ind] for ind in ind_top])
            scores.append(t_score)
            emosequence.append(t_score[2])  # 获取每个短句预测出来的概率最大的emoji
            print(t_score)
        return emosequence, TEST_SENTENCES_CHI

if __name__ == '__main__':
    ter = TERDeepMoji()
    emojilist, short_sentences = ter.textEmo("不要这样")
    print(emojilist, short_sentences)
