from __future__ import unicode_literals, print_function, division
from io import open

import random

import torch
import torch.nn as nn
from torch import optim
from Util import readFileList
import os
import pandas as pd
import numpy as np

torch.backends.cudnn.enabled = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(torch.nn.Module):
    def __init__(self,input_size,hidden_size):
        super(EncoderRNN,self).__init__()
        self.hidden_size = hidden_size

        self.embedding = torch.nn.Embedding(input_size,hidden_size)
        self.gru = torch.nn.GRU(hidden_size,hidden_size,bidirectional=True)

    def forward(self,input,hidden):
        #print("INPUT",input.size(),input)
        embeded = self.embedding(input).view(1,1,-1)
        output = embeded
        output,hidden = self.gru(output,hidden)
        return output,hidden

    def initHidden(self):
        return torch.zeros(2,1,self.hidden_size,device=device)

class DecoderRNN(torch.nn.Module):
    def __init__(self,hidden_size,output_size):
        super(DecoderRNN,self).__init__()
        self.hidden_size = hidden_size

        self.embedding = torch.nn.Embedding(output_size,hidden_size)
        self.gru = torch.nn.GRU(hidden_size,hidden_size,bidirectional=True)
        self.out = torch.nn.Linear(2*hidden_size,output_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self,input,hidden):
        #print("DecoderSize", input.size(), hidden.size())
        output = self.embedding(input).view(1,1,-1)
        output = torch.relu(output)

        #print("OUTPUT",output.size())
        output,hidden = self.gru(output, hidden)
        #print("Hidden", hidden.size())
        output = self.softmax(self.out(output[0]))

        return output,hidden

    def initHidden(self):
        return torch.zeros(1,1,self.hidden_size,device=device)


MAX_LENGTH = 30#한마디의 최대길이

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer,criterion,max_length = MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0
    #print("target tensor", target_tensor)

    for ei in range(input_length):
        encoder_output,encoder_hidden = encoder(input_tensor[ei],encoder_hidden)
        #encoder_outputs[ei] = encoder_output[0,0][:WordvectDimention]

    decoder_input = torch.tensor([[SOS_token]],device=device)
    #decoder_input = decoder_input.cuda()
    decoder_input = decoder_input.to(device)

    #decoder_hidden = encoder_hidden[0].unsqueeze(0)
    decoder_hidden = encoder_hidden
    #print("HiddenSize", encoder_hidden.size())
    decoder_hidden = decoder_hidden.to(device)

    use_tencher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_tencher_forcing:
        for di in range(target_length):
            #print("shit?")
            #decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden[0].unsqueeze(0))
            #print("decoder_hidden_size", decoder_hidden.size())
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)  # 확률과 one-hot
            #print(topi.item(), "and", target_tensor[di].item())

            #loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di] #teacher forcing
    else:
        for di in range(target_length):
            #decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden[0].unsqueeze(0))
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv,topi = decoder_output.topk(1) #확률 과 one-hot
            decoder_input = topi.squeeze().detach()

            #loss += criterion(decoder_output,target_tensor[di].unsqueeze(0))
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    #print(loss)
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item() / target_length

def saveAllmodel(DataSetName, name, encoder1, decoder1):
    SPATH = "Model\\{}\\Sgenerator".format(DataSetName) + name
    EPATH = "Model\\{}\\DecoderRNN".format(DataSetName) + name
    torch.save(encoder1, SPATH)
    torch.save(decoder1, EPATH)


def prepareData(FileHead, Filenumber):

    # 실험을 위한 데이터
    #filename = FileHead + str(Filenumber) + ".txt"
    #lines = open(filename, encoding='utf-8').read().split('\n')

    #학습시키기 위한 데이터
    data = pd.read_csv("EMSEData.csv")  # EMSEData
    data = data.dropna()
    check_nan_in_data = data.isnull().values.any()
    # print(check_nan_in_data)
    lines = list(np.array(data['Sentence'].tolist()))

    intsen = []
    intsvclen = []

    lineNum = len(lines)

    word2idx = {}
    word2cnt = {}
    idx2word = {0: "SOS", 1: "EOS"}
    n_words = 2

    for linenumber in range(lineNum):

        line = lines[linenumber]
        line = line.split()

        for sig in line:
            if sig not in word2idx:
                word2idx[sig] = n_words
                word2cnt[sig] = 1
                idx2word[n_words] = sig
                n_words += 1
            else:
                word2cnt[sig] += 1



    for linenumber in range(lineNum):
        # if linenumber%2==0:
        #     continue
        line = lines[linenumber]
        line = line.split()
        imptline = []

        # 단어 임베딩을 하려는 듯?
        for sig in line:
            imptline.append(word2idx[sig])

        if (len(imptline)==0):
            continue

        intsen.append(imptline.copy())
        intsvclen.append(len(imptline))

    print("vocab_size", n_words)
    sennum = len(intsen)

    return intsen, intsvclen, sennum

#def trainIters(encoder, decoder, n_iters, FileHead, FileNumm, learning_rate =0.01):
def trainIters(encoder, decoder, n_iters, DatasetName, learning_rate=0.01):
    #FileHead = "Vec\\{}\\Vec".format(DatasetName)
    FileHead = "bug report\\"
    #FileNList = readFileList(DatasetName)
    FileNList = [1]
    print("file list", FileNList)

    FN = len(FileNList)
    encoder_optimizer = optim.SGD(encoder.parameters(),lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(),lr=learning_rate)

    criterion = nn.NLLLoss()
    for iter in range(1,n_iters + 1):
        print("iter", iter)
        try:
            #num = random.randint(1, FileNumm + 1)
            #num = random.randint(0, FN-1)
            for num in FileNList:
            #num = FileNList[num]
                intcom, intsveclen, sennum = prepareData(FileHead, num)
                #print("intcom", intcom)
                targetcom = intcom.copy()
                a_loss = 0

                for senk in range(sennum):
                    input_tensor = torch.tensor(intcom[senk][:intsveclen[senk]].copy(), dtype=torch.long,device=device).view(-1, 1)
                    target_tensor = torch.tensor(targetcom[senk][:intsveclen[senk]].copy(), dtype=torch.long,device=device).view(-1, 1)

                    input_tensor = input_tensor.to(device)
                    target_tensor = target_tensor.to(device)

                    loss = train(input_tensor,target_tensor,encoder,decoder,encoder_optimizer,decoder_optimizer,criterion)
                    a_loss += loss
                print(iter, num, a_loss/sennum)
        except FileNotFoundError:
            continue
    return encoder, decoder

WordvectDimentionList = [500,600,700,800,900,1000,1200]
#WordvectDimentionList = [1200]

SOS_token = 0
EOS_token = 1

##filter
MAX_LENGTH = 30#한마디의 최대길이

teacher_forcing_ratio = 0.5

def trainEDcoder(WordvectDimentionList, DName):
    if (not os.path.exists("Model\\{}".format(DName))):
        os.makedirs("Model\\{}".format(DName))
    for k in WordvectDimentionList:
        print("Processing K:", k)
        WordvectDimention = k

        # for subjective judgement
        hidden_size = WordvectDimention
        encoder1 = EncoderRNN(7135, hidden_size).to(device)
        decoder1 = DecoderRNN(hidden_size, 7135).to(device)
        encoder1, decoder1 = trainIters(encoder1, decoder1, 50, DName)
        saveAllmodel(DName, str(WordvectDimention), encoder1, decoder1)



trainEDcoder(WordvectDimentionList,'procEMSESenVec')

# encoder = torch.load("Model\\procEMSESenVec\\Sgenerator1200")
# print(encoder.eval())