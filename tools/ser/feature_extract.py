import tools.ser.opensmile
import numpy as np
max_audio_length = 750


def extract_features_from(wav_file):
    mfcc = tools.ser.opensmile.get_mfcc_feature(wav_file)
    prosody = tools.ser.opensmile.get_prosodic_feature(wav_file)
    new_line = np.concatenate([prosody, np.zeros(len(mfcc[0]) - len(prosody))])
    if len(mfcc) > max_audio_length:
        mfcc = mfcc[-1 * max_audio_length:]
    return np.r_[mfcc, [new_line]]


if __name__ == '__main__':
    f = extract_features_from('Ses01F_impro02_F000.wav')
    print(f.shape)
