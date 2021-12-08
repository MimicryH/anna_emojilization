import os
import pandas as pd
import csv

opensmile_root = 'D:/DoobiePJ/2020Ali/SER_test/opensmile-3.0-win-x64/'
opensmile_path = opensmile_root + 'bin/SMILExtract.exe'
# 项目路径
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
# single_feature.csv 路径
single_feat_path = os.path.join(BASE_DIR, 'single_feature.csv')
# Opensmile 配置文件路径
mfcc_config_path = opensmile_root + 'config/mfcc/MFCC12_0_D_A.conf'
prosody_config_path = opensmile_root + 'config/prosody/prosodyAcf.conf'


def get_mfcc_feature(wav_path):
    cmd = opensmile_path + ' -C ' + mfcc_config_path +\
                                   ' -I ' + wav_path + \
                                   ' -csvoutput ' + single_feat_path
    print("Opensmile cmd: ", cmd)
    if os.system(cmd) == 0:
        df_mfcc = pd.read_csv(single_feat_path, ';').iloc[:, 2:-1]
    else:
        print('Opensmile cmd failed !!!!!!')
    # print(df_mfcc.shape)
    return df_mfcc.to_numpy()


def get_prosodic_feature(wav_path):
    config_path = prosody_config_path
    cmd = opensmile_path + ' -C ' + config_path +\
                                   ' -I ' + wav_path + \
                                   ' -csvoutput ' + single_feat_path
    print("Opensmile cmd: ", cmd)
    if os.system(cmd) == 0:
        csv_file = open(single_feat_path, 'r')
        reader = csv.reader(csv_file, delimiter=';')
        rows = [row for row in reader]
        last_line = rows[-1]
        return last_line[2:]
    else:
        print('Opensmile cmd failed !!!!!!')



if __name__ == '__main__':
    # r = get_opensmile_feature('D:\DoobiePJ/2020Ali\SER_test\opensmile-3.0-win-x64/example-audio/media-interpretation.wav')
    # print(r.shape)
    r = get_prosodic_feature('D:\DoobiePJ/2020Ali\SER_test\opensmile-3.0-win-x64/example-audio/media-interpretation.wav')
    print(len(r))