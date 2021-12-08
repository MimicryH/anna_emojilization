from tools.ser.feature_extract import extract_features_from
from tools.ser.Yoon_model import YoonBREAudio, YoonBREText, YoonBRE2TA
import torch
from torchtext import data
import numpy as np
from nltk.tokenize import TweetTokenizer
import os.path

label_dict = {
    "neu": 0,
    "ang": 1,
    "sad": 2,
    'hap': 3
}
MAX_TEXT_LEN = 128

vector_path = 'D:/DoobiePJ/2020Ali/SER_test/.vector_cache/'
cwd = os.path.split(os.path.realpath(__file__))[0]
x_t = np.load(cwd + '/ser/x_text.npy', allow_pickle=True)
TEXT = data.Field()
TEXT.build_vocab(x_t, max_size=25000, vectors='glove.6B.100d', vectors_cache=vector_path, unk_init=torch.Tensor.normal_)


tknzr = TweetTokenizer()
# Create a Tokenizer with the default settings for English
# including punctuation rules and exceptions

# model_t = YoonBREText(num_class=4, vocab_size=len(TEXT.vocab), embedding_dim=100,
#                       pad_idx=TEXT.vocab.stoi[TEXT.pad_token], hidden_size=128, num_layers=2, dropout=.3,
#                       bidirectional=True, use_cuda=True)
# model_t = torch.load('Yoon_text.pth')

# model_at = YoonBRE2TA()
# model_at.load_state_dict(torch.load(cwd + '/ser/Yoon_audio_text.pth'))

model_a = torch.load(cwd + '/ser/Yoon_audio.pth')


def ser_audio(wav_file):
    # model_a = YoonBREAudio(num_class=4, mfcc_size=119, prosody_size=35, hidden_size=128, num_layers=2,
    #                        dropout=0.3).cuda()
    audio_feature = extract_features_from(wav_file).astype(np.float_)
    i = torch.tensor(np.reshape(audio_feature, (1, audio_feature.shape[0], audio_feature.shape[1]))).float().cuda()
    prediction_a = model_a(i)
    return prediction_a
#
#
# def ter(text):
#     tokens = tknzr.tokenize(text)
#     if len(tokens) > MAX_TEXT_LEN:
#         tokens = tokens[:MAX_TEXT_LEN]
#     else:
#         tokens = tokens + [TEXT.pad_token] * (MAX_TEXT_LEN - len(tokens))
#     idxs = []
#     for token in tokens:
#         idxs.append(TEXT.vocab.stoi[str(token)])
#     t = torch.tensor(idxs).cuda()
#     t = t.view(1, t.shape[0])
#     prediction_a = model_t(t)
#     return prediction_a


def t_s_er(wav_file, text):
    a_feature = np.array([extract_features_from(wav_file)]).astype('float32')
    # print(a_feature)
    a = torch.tensor(a_feature).cuda()
    tokens = tknzr.tokenize(text)
    if len(tokens) > MAX_TEXT_LEN:
        tokens = tokens[:MAX_TEXT_LEN]
    else:
        tokens = tokens + [TEXT.pad_token] * (MAX_TEXT_LEN - len(tokens))
    idxs = []
    for token in tokens:
        idxs.append(TEXT.vocab.stoi[str(token)])
    t = torch.tensor(idxs).cuda()
    t = t.view(1, t.shape[0])
    return model_at(a, t).tolist()[0]


def emotion_detect(wav_file):
    txt = ser.xunfei.asr(wav_file)
    en_txt = ser.translate.google_translate(txt)
    print("translated:", en_txt)
    r = t_s_er(wav_file, en_txt)
    return {'neu': r[0], 'ang': r[1], 'sad': r[2], 'hap': r[3]}


if __name__ == '__main__':
    # r = ser('Ses01F_impro02_F000.wav')
    # r = ter("I like this ball.")[0]
    # r = t_s_er('Ses01F_impro02_F000.wav', 'Did she get the mail? She saw my letter.')
    # print(emotion_detect('output.wav'))
    # path = 'D:\DoobiePJ/2020Ali/annotation_website/future_bot-master/audio\jiabei/'
    # print(os.listdir(path)).
    # results = []
    # for wav_file in os.listdir(path):
    #     results.append(ser_audio(path+wav_file)[0].tolist())
    #
    # for index, r in enumerate(results):
    #     print(index, list(label_dict.keys())[r.index(max(r))])
    # print(emotion_detect('D:/DoobiePJ/phd/组会2021/17.wav'))
    print(ser_audio('D:/DoobiePJ/phd/组会2021/17.wav'))
