# This script is used to do a training process of a seq2seq model
# The first argument is the numBatchesBF16 parameter to determine
# the minimum number of batches to execute in reduced precision.
# The second argument defines the Exponential Moving Average
# threshold.
# The third argument is the name of the file to save the weights
# of the trained network
# The fourth argument will be a log to keep the output of training.
# At this moment is not being used.
# The fifth argument defines the name of the fifo to communicate
# with the pintool to change between precisions. The name needs to be
# fifopipe
# The sixth parameter is the number of training epochs to be performed
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

import spacy
import numpy as np

import random
import math
import time
import sys

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

spacy_de = spacy.load('de')
spacy_en = spacy.load('en')


################################################

n = 5 # Parameter to calculate the EMA
flagAttachBF16 = False # This flag is used to change
                       # between BF16 or FP32
niterToTestEMABF16 = int(sys.argv[1])
ema_change = float(sys.argv[2])
output_trained_model = sys.argv[3] # Save model used
                                   # by pytorch to
                                   # do the test
log_file = sys.argv[4] # File to be generated
                       # containing the stdout
                       # during training
is_ema_calculated = True # Flag to calculate the
                         # EMA average by the first
                         # time
EMA = 0.0
EMAprev = 0.0

path = sys.argv[5]
# path = 'fifopipe'
def attachFMA_MP_FP32_WU_BN():
    fifo = open(path, 'w')
    fifo.write('B')
    fifo.close()
    time.sleep(20)

def detach():
    fifo = open(path, 'w')
    fifo.write('G')
    fifo.close()
    time.sleep(20)

def attachFMA_BF16():
    fifo = open(path, 'w')
    fifo.write('A')
    fifo.close()
    time.sleep(20)

# In seq2seq model there are not
# BN layers.
def attachFMA_BF16_FP32_WU_BN():
    fifo = open(path, 'w')
    fifo.write('E')
    fifo.close()
    time.sleep(20)

# Exponential Moving Average Calculation
# The first EMA value is calculated as the AVERAGE of the first n values
# n = 5
# alpha = 2/(n+1)
# EMAt = alpha x actualvalue + (1-alpha)xEMAt-1
def ema(n, actualValue, EMAprev):
    alpha = 2.0 / (float(n) + 1.0)
    EMAt = alpha * actualValue + (1-alpha)*EMAprev
    return EMAt

################################################


def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings
    """
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]


SRC = Field(tokenize = tokenize_de,
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True)

TRG = Field(tokenize = tokenize_en,
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True)

train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'),
                                                    fields = (SRC, TRG))


SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 256

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = BATCH_SIZE,
    device = device)

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        #src = [src len, batch size]
        embedded = self.dropout(self.embedding(src))
        #embedded = [src len, batch size, emb dim]
        outputs, hidden = self.rnn(embedded)
        #outputs = [src len, batch size, hid dim * num directions]
        #hidden = [n layers * num directions, batch size, hid dim]

        #hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        #outputs are always from the last layer

        #hidden [-2, :, : ] is the last of the forwards RNN
        #hidden [-1, :, : ] is the last of the backwards RNN

        #initial decoder hidden is final hidden state of the forwards and backwards
        #  encoder RNNs fed through a linear layer
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))

        #outputs = [src len, batch size, enc hid dim * 2]
        #hidden = [batch size, dec hid dim]
        return outputs, hidden



class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Parameter(torch.randn(dec_hid_dim))

    def forward(self, hidden, encoder_outputs):
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        #repeat encoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        #hidden = [batch size, src len, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2)))

        #energy = [batch size, src len, dec hid dim]

        energy = energy.permute(0, 2, 1)

        #energy = [batch size, dec hid dim, src len]

        #v = [dec hid dim]

        v = self.v.repeat(batch_size, 1).unsqueeze(1)

        #v = [batch size, 1, dec hid dim]

        attention = torch.bmm(v, energy).squeeze(1)

        #attention= [batch size, src len]

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)

        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):

        #input = [batch size]
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]

        input = input.unsqueeze(0)

        #input = [1, batch size]

        embedded = self.dropout(self.embedding(input))

        #embedded = [1, batch size, emb dim]

        a = self.attention(hidden, encoder_outputs)

        #a = [batch size, src len]

        a = a.unsqueeze(1)

        #a = [batch size, 1, src len]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        #encoder_outputs = [batch size, src len, enc hid dim * 2]

        weighted = torch.bmm(a, encoder_outputs)

        #weighted = [batch size, 1, enc hid dim * 2]

        weighted = weighted.permute(1, 0, 2)

        #weighted = [1, batch size, enc hid dim * 2]

        rnn_input = torch.cat((embedded, weighted), dim = 2)

        #rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        #output = [seq len, batch size, dec hid dim * n directions]
        #hidden = [n layers * n directions, batch size, dec hid dim]

        #seq len, n layers and n directions will always be 1 in this decoder, therefore:
        #output = [1, batch size, dec hid dim]
        #hidden = [1, batch size, dec hid dim]
        #this also means that output == hidden
        assert (output == hidden).all()

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))

        #prediction = [batch size, output dim]

        return prediction, hidden.squeeze(0)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time

        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        #encoder_outputs is all hidden states of the input sequence, back and forwards
        #hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src)

        #first input to the decoder is the <sos> tokens
        input = trg[0,:]

        for t in range(1, trg_len):
            #insert input token embedding, previous hidden state and all encoder hidden states
            #receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden, encoder_outputs)

            #place predictions in a tensor holding predictions for each token
            outputs[t] = output

            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            #get the highest predicted token from our predictions
            top1 = output.argmax(1)

            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 512
DEC_HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec, device).to(device)

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

model.apply(init_weights)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')
sys.stdout.flush()

optimizer = optim.Adam(model.parameters())

TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

def train(model, iterator, optimizer, criterion, clip, EMA, flagAttachBF16, is_ema_calculated, FP32Counter, BF16Counter):
    model.train()
    epoch_loss = 0
    training_loss = np.zeros(len(iterator))
    counterAttachBF16 = 0
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        optimizer.zero_grad()
        output = model(src, trg)
        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        training_loss[i] = loss.item()
        if((i==n) and (is_ema_calculated)):
            EMA = np.average(training_loss)
            is_ema_calculated = False
        else:
            if(i>n):
                EMAprev = EMA
                EMA = ema(n, loss.item(), EMAprev)
                if(flagAttachBF16 != True):
                    if((EMAprev-EMA) > ema_change):
                        flagAttachBF16 = True
                        attachFMA_BF16_FP32_WU_BN()
                    FP32Counter+=1
                else:
                    BF16Counter+=1
                    counterAttachBF16 = counterAttachBF16 + 1
                    if(counterAttachBF16 == niterToTestEMABF16):
                        if(EMAprev-EMA) > ema_change:
                            counterAttachBF16 = 0
                        else:
                            flagAttachBF16 = False
                            attachFMA_MP_FP32_WU_BN()
                            counterAttachBF16 = 0
        # To run just one batch uncomment the code below
        # return epoch_loss / len(iterator)
    return epoch_loss / len(iterator), EMA, flagAttachBF16, is_ema_calculated, FP32Counter, BF16Counter

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output = model(src, trg, 0) #turn off teacher forcing
            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

N_EPOCHS = int(sys.argv[6])
CLIP = 1

best_valid_loss = float('inf')
print('Number of batches by epoch ', len(train_iterator))

BF16Counter = 0
FP32Counter = 0

for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss, EMA, flagAttachBF16, is_ema_calculated, FP32Counter, BF16Counter = train(model, train_iterator, optimizer,
                                                                                         criterion, CLIP, EMA, flagAttachBF16,
                                                                                         is_ema_calculated, FP32Counter,
                                                                                         BF16Counter)
    valid_loss = evaluate(model, valid_iterator, criterion)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), output_trained_model)
    print(epoch+1,',',train_loss,',',math.exp(train_loss),',',valid_loss,',',math.exp(valid_loss),',',BF16Counter,',',FP32Counter)
    sys.stdout.flush()
print('Number of batches calculated using BF16:', BF16Counter)
print('Number of batches calculated using FP32:', FP32Counter)
print('Number of batches processed:', BF16Counter+FP32Counter)
sys.stdout.flush()
