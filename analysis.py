import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import normaltest
import scipy.stats as stats
import seaborn as sns

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

question_type = ['例行问题', '高级追问', '扩展追问(补充)', '联想追问(接续)', '联想追问(同类)', '联想追问(属性)',
                 '原因追问', '确认追问', '关键词指向追问', '感受追问']
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


def get_dict_va(y, v, a):
    d = {'question': y, 'v': v, 'a': a}
    df_question = pd.DataFrame(data=d)
    question_v = {}
    question_a = {}
    for index, qt in enumerate(question_type):
        question_v[qt] = df_question[df_question['question'] == index]['v']
        question_a[qt] = df_question[df_question['question'] == index]['a']
    return question_v, question_a


def plot_hist(y, v, a):
    question_v, question_a = get_dict_va(y, v, a)
    # df_question.to_excel('df_question.xlsx')
    plt.figure()
    for i, qt in enumerate(question_type):
        try:
            print("dagostino_test:  v")
            dagostino_test(question_v[qt])
            print("dagostino_test:  a")
            dagostino_test(question_a[qt])
        except ValueError:
            print("sample size might be too small")
        plt.subplot(10, 2, (i+1)*2-1)
        plt.xlabel('v')
        plt.ylabel('Probability of '+str(i))
        plt.hist(question_v[qt], bins='auto', density=True)
        plt.subplot(10, 2, (i+1)*2)
        plt.xlabel('a')
        plt.ylabel('Probability of '+str(i))
        plt.hist(question_a[qt], bins='auto', density=True)

    # f, p = stats.kruskal(question_v['扩展追问(补充)'], question_v['联想追问(接续)'], question_v['联想追问(同类)'],
    #                       question_v['联想追问(属性)'], question_v['原因追问'], question_v['确认追问'], question_v['关键词指向追问'])
    f, p = stats.kruskal(pd.concat([question_v['联想追问(接续)'], question_v['联想追问(同类)'], question_v['联想追问(属性)']]),
                         question_v['扩展追问(补充)'], question_v['确认追问'], question_v['原因追问'])
    print('question_v kruskal:', f, p)
    # f, p = stats.kruskal(question_a['扩展追问(补充)'], question_a['联想追问(接续)'], question_a['联想追问(同类)'],
    #     #                       question_a['联想追问(属性)'], question_a['原因追问'], question_a['确认追问'], question_a['关键词指向追问'])
    f, p = stats.kruskal(pd.concat([question_a['联想追问(接续)'], question_a['联想追问(同类)'], question_a['联想追问(属性)']]),
                         question_a['扩展追问(补充)'], question_a['确认追问'], question_a['原因追问'])

    print('question_a kruskal:', f, p)
    plt.show()


def plot_scatter(y, v, a):
    d = {'question': y, 'v': v, 'a': a}
    df_question = pd.DataFrame(data=d)
    plt.figure()
    question_va = {}
    palette = sns.color_palette('muted')
    for i1, qt in enumerate(question_type):
        question_va[qt] = df_question[df_question['question'] == i1]
        plt.subplot(5, 2, i1+1)
        plt.title(qt)
        plt.axis([-3, 3, -3, 3])
        plt.xticks(np.arange(-3, 3, 0.5))
        plt.yticks(np.arange(-3, 3, 0.5))
        plt.grid(True, linestyle='-.')
        ax = plt.gca()
        ax.set_aspect('equal')
        # plt.xlabel('Valence')
        # plt.ylabel('Arousal')
        # sns.palplot(sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95, reverse=True))
        points = {}
        for i2, row2 in question_va[qt].iterrows():
            if points.__contains__((row2['v'], row2['a'])):
                points[(row2['v'], row2['a'])] += 1
            else:
                points[(row2['v'], row2['a'])] = 1

        for key0, key1 in points:
            plt.scatter(key0, key1, s=points[(key0, key1)] * 15, c=tuple(palette[i1]))
    plt.show()


def plot_error_bar(y, v, a):
    question_v, question_a = get_dict_va(y, v, a)
    n = []
    v_mean = []
    a_mean = []
    v_std = []
    a_std = []
    for i, qt in enumerate(question_type):
        n.append(len(question_v[qt]))
        v_mean.append(np.mean(question_v[qt]))
        a_mean.append(np.mean(question_a[qt]))
        v_std.append(np.std(question_v[qt]))
        a_std.append(np.std(question_a[qt]))

    plt.subplot(2, 1, 1)
    plt.errorbar(question_type, v_mean, yerr=v_std / np.sqrt(n), fmt='.k')

    plt.subplot(2, 1, 2)
    plt.errorbar(question_type, a_mean, yerr=a_std / np.sqrt(n), fmt='.k')
    plt.show()


def plot_box(y, v, a):
    question_v, question_a = get_dict_va(y, v, a)
    plt.subplot(2, 1, 1)
    plt.boxplot(question_v.values(), labels=question_type)
    plt.subplot(2, 1, 2)
    plt.boxplot(question_a.values(), labels=question_type)
    plt.show()


def anova(y, v, a):
    question_v, question_a = get_dict_va(y, v, a)
    f, p = stats.kruskal(question_v['扩展追问(补充)'], question_v['关键词指向追问'])

    print(f, p)
    f, p = stats.kruskal(question_a['扩展追问(补充)'], question_a['关键词指向追问'])
    print(f, p)


# read .xlsx
basedir = os.path.dirname(os.path.abspath(__file__))
tabledir = os.path.join(basedir, 'xlsx/总表3.xlsx')
df = pd.read_excel(tabledir, "总表-赵静")
df2 = pd.read_excel(tabledir, "总表-谢熊")
df = pd.concat([df, df2], sort=False)
print(len(df))
# pre-processing
v_t = []
a_t = []
v_s = []
a_s = []
v_f = []
a_f = []
v_ali = []
a_ali = []
v_manual = []
a_manual = []
question = []
count = pd.Series(np.zeros(len(emoji_list)), index=emoji_list)
# c_map = pd.DataFrame(0, index=emoji_list, columns=question_type)

for index, row in df.iterrows():
    if row['Speaker'] == 'A':
        v_t.append(row['text_v'])
        a_t.append(row['text_a'])
        v_s.append(row['speech_v'])
        a_s.append(row['speech_a'])
        v_f.append(row['fusion_v'])
        a_f.append(row['fusion_a'])
        # v_ali.append(row['v_ali'])
        # a_ali.append(row['a_ali'])
        v_manual.append(row['average_V'])
        a_manual.append(row['average_A'])
        question.append(question_type.index(row['question_type']))
        # results = which_emoji(row['emo_text'])
        # get the c_map
        # for emoji in results:
        #     count[emoji] += 1
        #     c_map[row['question_type']][emoji] += 1

# c_map.to_excel('c_map.xlsx')

# plot_scatter(question, v_manual, a_manual)
# plot_box(question, v_manual, a_manual)
anova(question, v_manual, a_manual)
plot_error_bar(question, v_manual, a_manual)
