import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import normaltest
import scipy.stats as stats
import seaborn as sns


question_type = ['例行问题', '例行问题子问题', '高级追问', '扩展追问(补充)', '扩展追问(关键词缺失）', '扩展追问(原因）', '确认追问', '关键词指向追问']
emoji_list = [u'\U0001F602', u'\U0001F612', u'\U0001F629', u'\U0001F62D', u'\U0001F60D', u'\U0001F614', u'\U0001F44C',
              u'\U0001F60A', u'\u2764', u'\U0001F60F', u'\U0001F601', u'\U0001F3B6', u'\U0001F633', u'\U0001F4AF',
              u'\U0001F634', u'\U0001F60C', u'\u263A', u'\U0001F64C', u'\U0001F495', u'\U0001F611', u'\U0001F605',
              u'\U0001F64F', u'\U0001F615', u'\U0001F618', u'\u2665', u'\U0001F610', u'\U0001F481', u'\U0001F61E',
              u'\U0001F648', u'\U0001F62B', u'\u270C', u'\U0001F60E', u'\U0001F621', u'\U0001F44D', u'\U0001F625',
              u'\U0001F62A', u'\U0001F60B', u'\U0001F624', u'\U0001F91A', u'\U0001F637', u'\U0001F44F', u'\U0001F440',
              u'\U0001F52B', u'\U0001F623', u'\U0001F608', u'\U0001F613', u'\U0001F494', u'\U0001F49F', u'\U0001F3A7',
              u'\U0001F64A', u'\U0001F609', u'\U0001F480', u'\U0001F616', u'\U0001F604', u'\U0001F61C', u'\U0001F620',
              u'\U0001F645', u'\U0001F4AA', u'\U0001F44A', u'\U0001F49C', u'\U0001F496', u'\U0001F499', u'\U0001F62C',
              u'\u2728']
c_map = pd.DataFrame(0, index=emoji_list, columns=question_type)


def which_emoji(text):
    results = []
    for emoji in emoji_list:
        if emoji in text:
            results.append(emoji)
    return results

def dagostino_test(data):
    # normality test
    stat, p = normaltest(data)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')

basedir = os.path.dirname(os.path.abspath(__file__))
tabledir = os.path.join(basedir, 'xlsx/总表.xlsx')
df = pd.read_excel(tabledir, "工作表1")
df_null = df.isnull()
count = pd.Series(np.zeros(len(emoji_list)), index=emoji_list)

v = []
a = []
y = []

for index, row in df.iterrows():
    if not df_null['question_type'][index]:
        if row['Speaker'] == 'A':
            v.append(row['fusion_v'])
            a.append(row['fusion_a'])
            y.append(question_type.index(row['question_type']))
            results = which_emoji(row['emo_text'])
            for emoji in results:
                count[emoji] += 1
                c_map[row['question_type']][emoji] += 1
print(count)

for index, row in c_map.iterrows():
    for q in question_type:
        if count[index] > 0:
            print(row[q], "/", count[index])
            print(float(row[q]/count[index]))
            c_map.loc[index, q] = float(row[q]/count[index])

print(c_map)


def plot_hist():
    d = {'question': y, 'v': v, 'a': a}
    df_question = pd.DataFrame(data=d)
    print(df_question)
    question_v = {}
    question_a = {}
    # df_question.to_excel('df_question.xlsx')
    plt.figure()
    for index, qt in enumerate(question_type):
        print(index, qt)
        question_v[qt] = df_question[df_question['question'] == index]['v']
        question_a[qt] = df_question[df_question['question'] == index]['a']
        try:
            print("dagostino_test:  v")
            dagostino_test(question_v[qt])
            print("dagostino_test:  a")
            dagostino_test(question_a[qt])
        except ValueError:
            print("sample size might be too small")
        plt.subplot(8, 2, (index+1)*2-1)
        plt.xlabel('v')
        plt.ylabel('Probability of '+str(index))
        plt.hist(question_v[qt], bins='auto', density=True)
        plt.subplot(8, 2, (index+1)*2)
        plt.xlabel('a')
        plt.ylabel('Probability of '+str(index))
        plt.hist(question_a[qt], bins='auto', density=True)

    f, p = stats.kruskal(question_v['扩展追问(补充)'], question_v['扩展追问(关键词缺失）'], question_v['扩展追问(原因）'],
                          question_v['确认追问'], question_v['关键词指向追问'])
    print('question_v kruskal:', f, p)
    f, p = stats.kruskal(question_a['扩展追问(补充)'], question_a['扩展追问(关键词缺失）'], question_a['扩展追问(原因）'],
                          question_a['确认追问'], question_a['关键词指向追问'])
    print('question_a kruskal:', f, p)
    c_map.to_excel('c_map.xlsx')
    plt.show()


def plot_scatter():
    d = {'question': y, 'v': v, 'a': a}
    df_question = pd.DataFrame(data=d)
    plt.figure()
    question_va = {}
    palette = sns.color_palette('Paired')
    for i1, qt in enumerate(question_type):
        question_va[qt] = df_question[df_question['question'] == i1]
        plt.subplot(4, 2, i1+1)
        plt.title("question type " + str(i1+1))
        plt.axis([-3, 3, -3, 3])
        plt.xticks(np.arange(-3, 3, 0.5))
        plt.yticks(np.arange(-3, 3, 0.5))
        plt.grid(True, linestyle='-.')
        ax = plt.gca()
        ax.set_aspect('equal')
        plt.xlabel('Valence')
        plt.ylabel('Arousal')
        # sns.palplot(sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95, reverse=True))
        for i2, row2 in question_va[qt].iterrows():
            plt.scatter(row2['v'], row2['a'], c=tuple(palette[i1]))
    plt.show()

# plot_hist()
plot_scatter()