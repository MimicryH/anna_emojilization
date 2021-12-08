
import torch

import torch.nn.functional as F
# import xunfei

print(torch.__version__)


class YoonBRE(torch.nn.Module):
    def __init__(self, num_class, hidden_size, num_layers, dropout, bidirectional=True, use_cuda=True):
        super(YoonBRE, self).__init__()
        self.num_class = num_class
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.use_cuda = use_cuda


class YoonBREAudio(YoonBRE):
    def __init__(self, num_class, mfcc_size, prosody_size, hidden_size, num_layers, dropout, bidirectional=True, use_cuda=True):
        super(YoonBREAudio, self).__init__(num_class, hidden_size, num_layers, dropout, bidirectional, use_cuda)
        self.rnn = torch.nn.GRU(mfcc_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = torch.nn.Linear(hidden_size * 2 + prosody_size, num_class)
        self.mfcc_size = mfcc_size
        self.prosody_size = prosody_size

    def forward(self, i_audio):
        # inputs shape (batch, max_audio_length + 1, input_size)
        # inputs shape (64, 750 + 1, 121)
        _, hidden = self.rnn(i_audio[:, :-1, :self.mfcc_size])
        # out shape (750, 64, 256)  (seq_len, batch, num_directions * hidden_size)
        # hidden shape (4, 64, 128) (num_layers * num_directions, batch, hidden_size)
        hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)
        out = self.fc(torch.cat([hidden, i_audio[:, -1, :self.prosody_size]], dim=1))
        out = F.softmax(out, 1)
        return out


class YoonBREText(YoonBRE):
    def __init__(self, num_class, vocab_size, embedding_dim, pad_idx, hidden_size, num_layers, dropout,
                 bidirectional=True, use_cuda=True):
        super(YoonBREText, self).__init__(num_class, hidden_size, num_layers, dropout, bidirectional, use_cuda)
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = torch.nn.GRU(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=dropout,
                                bidirectional=True)
        self.fc = torch.nn.Linear(hidden_size * 2, num_class)

    def forward(self, i_text):
        # inputs shape (batch, max_audio_length + 1, input_size)
        # inputs shape (64, 750 + 1, 121)
        embedded = self.embedding(i_text)
        _, hidden = self.rnn(embedded)
        # out shape (750, 64, 256)  (seq_len, batch, num_directions * hidden_size)
        # hidden shape (4, 64, 128) (num_layers * num_directions, batch, hidden_size)
        hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)
        out = self.fc(hidden)
        out = F.softmax(out, 1)
        return out


class YoonBRE2TA(YoonBRE):
    def __init__(self, num_class, mfcc_size, prosody_size, vocab_size, embedding_dim, pad_idx, hidden_size, num_layers,
                 dropout, bidirectional=True, use_cuda=True):
        super(YoonBRE2TA, self).__init__(num_class, hidden_size, num_layers, dropout, bidirectional, use_cuda)
        self.rnn_a = torch.nn.GRU(mfcc_size, hidden_size, num_layers, batch_first=True, dropout=dropout,
                                  bidirectional=True)
        self.mfcc_size = mfcc_size
        self.prosody_size = prosody_size
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn_t = torch.nn.GRU(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=dropout,
                                  bidirectional=True)
        self.fc = torch.nn.Linear((hidden_size * 2 + prosody_size) * 2, num_class)

    def forward(self, i_audio, i_text):
        # inputs shape (batch, max_audio_length + 1, input_size)
        # inputs shape (64, 750 + 1, 121)
        batch_size = i_audio.size()[0]
        prosody_vec = i_audio[:, -1, :self.prosody_size].view(batch_size, 1, self.prosody_size)
        out_a, _ = self.rnn_a(i_audio[:, :-1, :self.mfcc_size])

        # expand the prosody_vec
        # cat p_v  (64, 750, 35) after out_a (64, 750, 256)
        p_v_a = prosody_vec.expand(batch_size, out_a.size()[1], self.prosody_size)
        O_A = torch.cat([out_a, p_v_a], dim=2)
        #
        O_A_last = O_A[:, -1, :]
        # O_A_last (64, 291)

        out_t, _ = self.rnn_t(self.embedding(i_text))
        # out_t = (64, 128, 256)

        # expand the prosody_vec
        # cat p_v  (64, 128, 35) after out_t (64, 128, 256)     O_T (64, 128, 291)
        p_v_t = prosody_vec.expand(batch_size, out_t.size()[1], self.prosody_size)
        O_T = torch.cat([out_t, p_v_t], dim=2)
        O_T_last = O_T[:, -1, :]

        # get A_1 (64, 128) sum_a_i_t (64)
        A_1 = torch.zeros([batch_size, O_T.size()[1]]).cuda()
        sum_a_i_t = torch.zeros(batch_size).cuda()
        for o_i in range(O_T.size()[1]):
            r = torch.sum(torch.mul(O_A_last, O_T[:, o_i]), dim=1)
            A_1[:, o_i].add_(r)
            sum_a_i_t.add_(r)

        A_1.mul_(1 / sum_a_i_t.expand(O_T.size()[1], -1).transpose(0, 1))
        A_1 = A_1.view(batch_size, A_1.size()[1], 1)
        H_1 = torch.sum(A_1.expand(batch_size, O_T.size()[1], O_T.size()[2]) * O_T, dim=1)
        # get A_2 (64, 750) sum_a_i_t (64)
        A_2 = torch.ones([batch_size, O_A.size()[1]]).cuda()
        sum_a_i_a = torch.ones(batch_size).cuda()
        for o_i in range(O_A.size()[1]):
            r = torch.sum(torch.mul(O_T_last, O_A[:, o_i]), dim=1)
            A_2[:, o_i].add_(r)
            sum_a_i_a.add_(r)
        # # sum_a_i_a (64, 750)
        A_2.mul_(1 / sum_a_i_a.expand(O_A.size()[1], -1).transpose(0, 1))

        A_2 = A_2.view(batch_size, A_2.size()[1], 1)
        H_2 = torch.sum(A_2.expand(-1, -1, O_A.size()[2]) * O_A, dim=1)
        return F.softmax(self.fc(torch.cat([H_1, H_2], dim=1)), 1)
