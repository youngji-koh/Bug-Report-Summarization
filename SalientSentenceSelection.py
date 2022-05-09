from DataPreprocess import DataInput, SigDataInput, DataPreprocess, BugSumSigDataInput, BugSumSigDataWithoutTestInput
from EvaluationBehaviorCapture import EvaluationBehaviorCapture
from Util import DataListList2float, DataListList2int, StrList2FloatList, StrList2IntList, DataListListProcess, wirteDataList, wordResCounter, wordResGoldenCounter, AccuracyMeasure, answerTypeTrans
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def BugSum(datasetName, beamSize, alpha, beta, gamma, sentenceWeight, sumlen):
    #DataPreprocess(datasetName)
    EvaluationBehaviorCapture(datasetName, alpha, beta, gamma, sentenceWeight)
    sheetNameList, dataList = DataInput(datasetName)
    sheetNum = len(sheetNameList)
    averageAcc = 0
    averageRecall = 0
    averageFscore = 0
    #BugSumWithoutTest(datasetName, beamSize)

    # SDS 에서 evaluated sentence 없는 것만 뽑은 것
    TEST = [9,10,16,17,18,20,21,24,25,29,30,33,36]

    # SDS
    #TEST = [9, 10, 17, 18, 20, 21, 25, 29, 30, 33, 36]

    EST = [3, 4, 5, 6, 12, 14, 15, 17, 18, 28, 37, 39, 41, 42, 44, 49, 51, 52, 53, 54, 55, 57, 59, 61, 62, 65, 66, 67,
           68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 95,
           96]    
    '''
    # ADS에서 w/o evaluated sentence 중 6개 뺀 거
    EST = [3, 4, 5, 6, 12, 14, 15, 17, 18, 28, 37, 39, 41, 42, 44, 49, 52, 54, 55, 57, 59, 61, 62, 65, 66, 67,
           68, 69, 70, 71, 72, 73, 75, 76, 77, 78, 80, 81, 82, 83, 84, 85, 86, 87, 88, 90, 91, 92, 95, 96]
    '''
    temp = 0
    numofBR = 0

    for i in range(sheetNum):

        # Evaluation 없는 것만
        if (i+1) not in EST:
            continue
        #print(i+1)
        numofBR = numofBR + 1


        #print("Processing", sheetNameList[i])
        senList, lemSenList, senNumberList, authorList, senInvolvecd, senVecList, senOPscore, senTRscore, senTPscore, senBRscore, senRank, tfidfWordList, tfidfScoreList, ignoreList, goldenList, evaluationList, evaluationTimeList, buildInfoMark, fscoreList = BugSumSigDataInput(dataList, i)

        # 요약문 길이 비율
        #wordRes = wordResCounter(ignoreList, 0.3)
        # ignore list 포함할 경우
        wordRes = len(senList) * sumlen

        #wordRes = round(len(senList) * 0.25)
        #print("Len", round(len(senList) * 0.3), '\t', "wordRes", wordRes - 1)

        #answer, loss = BeamSearchBert(senVecList, fscoreList, ignoreList, wordRes, senList, beamSize)

        # Bert 이용
        #answer, loss = BeamSearchBert(senVecList, fscoreList, ignoreList, buildInfoMark, wordRes, senList, beamSize)

        # Top Score
        answer = TopScore(senVecList, fscoreList, wordRes, ignoreList)
        #print("=================================")
        #print("BR." , numofBR)
        #print(answer)
        #print(goldenList)

        # Autoencoder 이용
        #MODELD = 1200 
        #answer, loss = BeamSearch(senVecList, ignoreList, fscoreList, Encoder, beamSize, wordRes, MODELD)

        #print("selectedSen", answer)
        #print(answer, loss)

        #print("goldenSen", list(filter(lambda x: goldenList[x] == 1, range(len(goldenList)))))
        acc, recall, f_score = AccuracyMeasure(answer, goldenList, wordRes)
        transedAnswer = answerTypeTrans(answer, senList)

        #print(transedAnswer)
        dataList[i]["BugSumSelected"] = transedAnswer
        dataList[i]["Performance"] = [acc, recall, f_score].copy()

        # default code
        averageAcc = averageAcc + acc
        averageRecall = averageRecall + recall
        averageFscore = averageFscore + f_score

        #print(i+1, f_score)


    #     x = [16, 18, 34, 39, 46, 47]
    #     # # Temporary code for ADS w/o evaluated sentences
    #     if i not in x:
    #         averageAcc = averageAcc + acc
    #         averageRecall = averageRecall + recall
    #         averageFscore = averageFscore + f_score
    #         temp = temp + 1
    #         print(i, acc, recall, f_score)
    #
    # print("temp", temp, "sheetNum", sheetNum, "EST", len(EST))

    wirteDataList(dataList, sheetNameList, datasetName)

    # temp
    # print("sheetNum:", sheetNum)
    # averageAcc = averageAcc/temp
    # averageRecall = averageRecall/temp
    # #averageFscore = 2 * averageAcc * averageRecall / (averageAcc + averageRecall)
    # averageFscore = averageFscore/temp

    # number of BugReport
    print("number of BR:", numofBR)
    averageAcc = averageAcc/numofBR
    averageRecall = averageRecall/numofBR
    #averageFscore = 2 * averageAcc * averageRecall / (averageAcc + averageRecall)
    averageFscore = averageFscore/numofBR

    # print("EST",len(EST))
    # averageAcc = averageAcc / len(EST)
    # averageRecall = averageRecall/len(EST)
    # #averageFscore = 2 * averageAcc * averageRecall / (averageAcc + averageRecall)
    # averageFscore = averageFscore/len(EST)


    # default
    # averageAcc = averageAcc/sheetNum
    # averageRecall = averageRecall/sheetNum
    # #averageFscore = 2 * averageAcc * averageRecall / (averageAcc + averageRecall)
    # averageFscore = averageFscore/sheetNum


    return averageAcc, averageRecall, averageFscore


def BugSumWithoutTest(datasetName, beamSize):
    #DataPreprocess(datasetName)
    #EvaluationBehaviorCapture(datasetName)
    sheetNameList, dataList = DataInput(datasetName)
    sheetNum = len(sheetNameList)

    for i in range(sheetNum):
        #print("Processing", sheetNameList[i])
        senList, lemSenList, senNumberList, authorList, senInvolvecd, senVecList, senOPscore, senTRscore, senTPscore, tfidfWordList, tfidfScoreList, ignoreList, evaluationList, evaluationTimeList, buildInfoMark, fscoreList = BugSumSigDataWithoutTestInput(dataList, i)
        wordRes = wordResCounter(ignoreList, 0.3)
        #print("wordRes", wordRes)
        #answer, loss = BeamSearchBert(senVecList, fscoreList, ignoreList, wordRes, senList, beamSize)
        answer, loss = BeamSearchBert(senVecList, fscoreList, ignoreList, buildInfoMark, wordRes, senList, beamSize)
        #print("selectedSen", answer)
        transedAnswer = answerTypeTrans(answer, senList)

        dataList[i]["BugSumSelected"] = transedAnswer
        #dataList[i]["Performance"] = [acc, recall].copy()
    wirteDataList(dataList, sheetNameList, datasetName)


def TopScore(senVecList, fscore, wordRes, ignoreList):
    # WeightedDataVec = VecMulFscore(senVecList, fscore)
    # FTVec = GenFullTVec(WeightedDataVec, ignoreList)
    #
    # senVecList = np.array(senVecList)
    # FTVec = np.array(FTVec)
    #
    # embedding_dim = len(FTVec)
    # SSM = np.zeros([len(senVecList)])
    #
    # for i in range(len(senVecList)):
    #     SSM[i] = cosine_similarity(senVecList[i].reshape(1, embedding_dim),
    #                                        FTVec.reshape(1, embedding_dim))[0, 0]
    #
    # senrank = sorted(SSM, reverse=True)
    # print(SSM)
    # answer = []
    #
    # for i in range(len(SSM)):
    #     # print(r, senTRscore.index(r))
    #     answer.append(list(SSM).index(senrank[i]))
    #     #print("scrore", senrank[i])
    #     #print("index", list(SSM).index(senrank[i]))
    #     # print(i, answer)
    #     if len(answer) >= wordRes:
    #         break


    # 기존 방법
    import random

    answer = []
    ranked_scores = sorted(fscore, reverse=True)

    if max(ranked_scores) == 1:
        for i in range(len(fscore)):
            randomIndex = random.randrange(0, len(fscore))
            while randomIndex in answer:
                randomIndex = random.randrange(0, len(fscore))
            answer.append(randomIndex)
            if len(answer) >= wordRes:
                break


    else:
        # print("ranked")
        # print(ranked_scores)

        for i in range(len(fscore)):
            answer.append(fscore.index(ranked_scores[i]))
            # print("score", ranked_scores[i])
            # print("index", fscore.index(ranked_scores[i]))
            if len(answer) >= wordRes:
                break

    return answer


def BeamSearchBert(senVecList, fscore, ignoreList, buildInfoMark, wordRes, senList, beamSize):
    #print("ignoreList", ignoreList)

    senNum = len(senVecList)

    Lnew = []
    LnewLoss = []
    Lold = []
    Chosen = []
    ChosenLoss = []

    Nvec = []
    #Ovec = []

    imptlist = []

    for i in range(senNum):
        # ingoreList 문장 포함하기 (모든 문장은 중요하다) (아래 두문장을 주석 처리 할 경우)
        # if (ignoreList[i] == 1):
        #     continue

        imptlist.clear()
        imptlist.append(i)
        Lold.append(imptlist.copy())
        #Ovec.append(zeroT.copy())

    #print("senVecList", senVecList[0])
    WeightedDataVec = VecMulFscore(senVecList, fscore)

    FTVec = GenFullTVec(WeightedDataVec, ignoreList)

    while (len(Lold) > 0):
        # print("Lold", Lold)
        LoldNum = len(Lold)
        for candiListNumber in range(LoldNum):
            candiList = Lold[candiListNumber]
            # print("candiList", candiList)
            for i in range(senNum):
                # 모든 문장은 중요하다!
                if (ignoreList[i] == 1):
                    continue
                if i not in candiList:
                    newCandi = candiList.copy()
                    newCandi.append(i)
                    # print("newCandi", newCandi)
                    flag = 1
                    for kList in Lnew:
                        if (ListCmp(kList, newCandi)):
                            flag = 0
                            break
                    if (flag):
                        # if (SentenceNumLengthCheck(sentencelist, newCandi, 2*wordRes)):
                        #ReconDvec = ReconsDvec(Svecs_SVM, newCandi, FcoreList, MODELD)
                        ReconFTvec = ReconFullTVec(senVecList, newCandi)

                        # ReconDvec = ReconDvec.cpu().numpy().tolist()[0]
                        # loss = CosineSimiarlty(ReconDvec, Dvec)
                        # print("Loss", loss)

                        loss = BertVecLoss(FTVec, ReconFTvec)
                        # print("start")
                        # infroincrease = criterion(ReconDvec, Dvec)
                        # Novelties = novelty(candiListNumber, Ovec, newCandi, SvecList, FcoreList, 0.2)
                        # Novelties = criterion()
                        # loss = infroincrease + Novelties
                        # loss = criterion(ReconDvec, Dvec) + novelty(candiListNumber, Ovec, newCandi, SvecList, FcoreList, 0.1)
                        # print("end")

                        # Lnew, LnewLoss, Chosen, ChosenLoss = appendLnew(newCandi, loss, Lnew, LnewLoss, Chosen, ChosenLoss, sentencelist, wordRes, beamSize)
                        Lnew, LnewLoss, Chosen, ChosenLoss, Nvec = appendLnew(newCandi, loss, Lnew, LnewLoss, Chosen, ChosenLoss, senList, wordRes, beamSize, Nvec, ReconFTvec)
            '''
            print("Lnew", Lnew)
            print("LnewLoss", LnewLoss)
            print("Lold", Lold)
            print("Chosen", Chosen)
            print("ChosenLoss", ChosenLoss)
            '''
        # print("Lnew", Lnew)
        # print("LnewLoss", LnewLoss)
        # print("Lold", Lold)
        Lold = Lnew.copy()
        Ovec = Nvec.copy()
        Lnew.clear()
        LnewLoss.clear()
    answer = []
    loss = 0
    if (wordRes == 1):
        for i in range(senNum):
            # 모든 문장은 중요하다!
            if (ignoreList[i] == 1):
                continue
            answer.append(i)
    else:
        #print("ChosenLossSize", len(Chosen), wordRes)
        pos, loss = locateSamllestOne(ChosenLoss)
        answer = Chosen[pos]

    # print(pos, loss)
    return answer, loss
'''
def wordResCounter(ignoreList, goldenList, percent, goldenPercent):
    senNum = len(ignoreList)
    goldenSenNum = len(goldenList)
    counter = 0
    for i in range(senNum):
        if (ignoreList[i]==0):
            counter = counter + 1
    answer = int(counter*percent)
    if answer == 0:
        answer = 1

    goldenCounter = 0
    for i in range(goldenSenNum):
        if (goldenList[i]==1):
            goldenCounter = goldenCounter + 1
    goldenAnswer = int(goldenCounter*goldenPercent)
    if goldenAnswer>counter:
        goldenAnswer = counter
    if goldenAnswer == 0:
        goldenAnswer = 1
    return answer, goldenAnswer
'''




def BertVecLoss(Veca, Vecb):
    #print("Veca", len(Veca))
    #print("Vecb", len(Vecb))
    VecLen = len(Veca)
    lossAnswer = 0.0
    for i in range(VecLen):
        lossAnswer = lossAnswer + (Veca[i]-Vecb[i])**2
    return lossAnswer

def VecAdd(Veca, Vecb):
    VecLen = len(Veca)
    answerList = []
    for i in range(VecLen):
        answerList.append(Veca[i]+Vecb[i])
    return answerList

def VecMulNumber(Veca, number):
    VecLen = len(Veca)
    answerList = []
    #print("Veca", Veca)
    for i in range(VecLen):
        #print("Veca[i]", Veca[i])
        answerList.append(Veca[i]*number)
    return answerList

def VecMulFscore(senVecList, senOPscore):
    senNum = len(senVecList)
    newSenVecList = []
    for i in range(senNum):
        #print("senVecList[i]", senVecList[i][0])
        imptList = VecMulNumber(senVecList[i], senOPscore[i])
        newSenVecList.append(imptList.copy())
    return newSenVecList

def GenFullTVec(senVecList, buildInfoMark):
    vecLen = len(senVecList[0])
    FullTVec = []
    for i in range(vecLen):
        FullTVec.append(0)
    senVecListNum = len(senVecList)
    for inum in range(senVecListNum):
        i=senVecList[inum]
        if (buildInfoMark[inum]!=1):
            FullTVec = VecAdd(FullTVec, i)
    return FullTVec

def ReconFullTVec(senVecList, chosenList):
    FullTVec = []
    vecLen = len(senVecList[0])
    for i in range(vecLen):
        FullTVec.append(0)
    for i in chosenList:
        FullTVec = VecAdd(FullTVec, senVecList[i])
    return FullTVec





def BeamSearch(sentencelist, avoidSentenceList, FcoreList, Encoder, beamSize, wordRes, MODELD):
    #print("avoidSentenceList", avoidSentenceList)
    #print("wordRes", wordRes)
    with torch.no_grad():#이건 역 전파 취소야
        criterion = torch.nn.MSELoss()
        #criterion = CosineSimiarlty()
        sennum = len(sentencelist)
        Lnew = []
        LnewLoss = []
        Lold = []
        Chosen = []
        ChosenLoss =[]

        Nvec = []
        Ovec = []

        imptlist = []
        zeroT = zeroTensor(2 * MODELD)

        for i in range(sennum):
            if (avoidSentenceList[i]==1):
                continue
            imptlist.clear()
            imptlist.append(i)
            Lold.append(imptlist.copy())
            Ovec.append(zeroT.copy())

        #print("sentencelist", sentencelist)

        Svecs_SVM, SvecList = AverageDvecGenProcess(sentencelist, FcoreList, sennum, Encoder, MODELD)

        #chosenlist = selectAll(sennum)

        chosenlist = reverseAvoid(avoidSentenceList)

        #Dvec = ReconsDvec(Svecs_SVM, chosenlist, FcoreList)
        Dvec = DvecGen(Svecs_SVM, chosenlist, FcoreList, MODELD)
        #Dvec = Dvec.cpu().numpy().tolist()[0]

        while(len(Lold)>0):
            #print("Lold", Lold)
            LoldNum = len(Lold)
            for candiListNumber in range(LoldNum):
                candiList = Lold[candiListNumber]
                #print("candiList", candiList)
                for i in range(sennum):
                    if (avoidSentenceList[i]==1):
                        continue
                    if i not in candiList:
                        newCandi = candiList.copy()
                        newCandi.append(i)
                        #print("newCandi", newCandi)
                        flag = 1
                        for kList in Lnew:
                            if (ListCmp(kList, newCandi)):
                                flag = 0
                                break
                        if (flag):
                            #if (SentenceNumLengthCheck(sentencelist, newCandi, 2*wordRes)):
                            ReconDvec = ReconsDvec(Svecs_SVM, newCandi, FcoreList, MODELD)

                            #ReconDvec = ReconDvec.cpu().numpy().tolist()[0]
                            #loss = CosineSimiarlty(ReconDvec, Dvec)
                            #print("Loss", loss)

                            loss = criterion(ReconDvec, Dvec)
                            #print("start")
                            #infroincrease = criterion(ReconDvec, Dvec)
                            #Novelties = novelty(candiListNumber, Ovec, newCandi, SvecList, FcoreList, 0.2)
                            #Novelties = criterion()
                            #loss = infroincrease + Novelties
                            #loss = criterion(ReconDvec, Dvec) + novelty(candiListNumber, Ovec, newCandi, SvecList, FcoreList, 0.1)
                            #print("end")

                            #Lnew, LnewLoss, Chosen, ChosenLoss = appendLnew(newCandi, loss, Lnew, LnewLoss, Chosen, ChosenLoss, sentencelist, wordRes, beamSize)
                            Lnew, LnewLoss, Chosen, ChosenLoss, Nvec = appendLnew(newCandi, loss, Lnew, LnewLoss, Chosen, ChosenLoss, sentencelist, wordRes, beamSize, Nvec, ReconDvec)
                #print("Lnew", Lnew)
                #print("LnewLoss", LnewLoss)
                #print("Lold", Lold)
                #print("Chosen", Chosen)
                #print("ChosenLoss", ChosenLoss)
            #print("Lnew", Lnew)
            #print("LnewLoss", LnewLoss)
            #print("Lold", Lold)
            Lold = Lnew.copy()
            Ovec = Nvec.copy()
            Lnew.clear()
            LnewLoss.clear()
        answer = []
        loss = 0
        if(wordRes == 1):
            for i in range(sennum):
                if (avoidSentenceList[i] == 1):
                    continue
                answer.append(i)
        else:
            #print("ChosenLossSize", len(Chosen), wordRes)
            pos, loss = locateSamllestOne(ChosenLoss)
            answer = Chosen[pos]

        #print(pos, loss)
        return answer, loss

def reverseAvoid(senList):
    senNum = len(senList)
    newSenList = []
    for i in range(senNum):
        if senList[i] == 0:
            newSenList.append(i)
    return newSenList

def DvecGen(Svecs_SVM, chosenlist, intsvm, MODELD):
    Dvec = torch.zeros((1, 2*MODELD))
    Dvec = Dvec.to(device)
    for num in chosenlist:
        Dvec = torch.add(Dvec, Svecs_SVM[num])

    '''total = 0
    for i in chosenlist:
        total = total + intsvm[i]'''

    #average_sennum = 1.0/total

    #average_sennum = 1.0/len(chosenlist)

    average_sennum = 1.0

    average_sennum = torch.tensor(average_sennum, dtype=torch.float, device=device)
    average_sennum = average_sennum.to(device)
    Dvec = torch.mul(Dvec, average_sennum)
    return Dvec

def ReconsDvec(Svecs_SVM, chosenlist, intsvm, MODELD):
    Dvec = torch.zeros((1, 2*MODELD))
    Dvec = Dvec.to(device)
    for num in chosenlist:
        Dvec = torch.add(Dvec, Svecs_SVM[num])

    '''total = 0
    for i in chosenlist:
        total = total + intsvm[i]

    average_sennum = 1.0/total'''

    average_sennum = 1.0/len(chosenlist)

    average_sennum = 1.0

    average_sennum = torch.tensor(average_sennum, dtype=torch.float, device=device)
    average_sennum = average_sennum.to(device)
    Dvec = torch.mul(Dvec, average_sennum)
    return Dvec

def ListCmp(list1, list2):
    if (len(list1)!=len(list2)):
        return False
    for itemk in list1:
        if itemk not in list2:
            return False
    return True

def locateSamllestOne(List):
    smallestNumber = 100
    smallestPos = 0
    listLength = len(List)
    for i in range(listLength):
        if List[i] < smallestNumber:
            smallestNumber = List[i]
            smallestPos = i
    return smallestPos, smallestNumber

def zeroTensor(vecSize):
    TenVec = []
    for i in range(vecSize):
        TenVec.append(0)

    #answer = torch.tensor(TenVec, dtype=torch.float, device=device)
    answer = TenVec
    return answer

def appendLnew(newCandi, loss, Lnew, LnewLoss, Chosen, ChosenLoss, sentencelist, wordNumRes, beamSize, Nvec, ReconVec):
    #print("Lnew", Lnew)
    #print("LnewLoss", LnewLoss)
    #print("newCandi", newCandi, loss)
    if loss<0:
        loss = loss*-1

    #ReconVec = ReconVec.cpu().numpy().tolist()[0]

    if extendable(sentencelist, newCandi, wordNumRes):
        if len(Lnew) < beamSize:
            Lnew.append(newCandi.copy())
            LnewLoss.append(loss)
            Nvec.append(ReconVec.copy())
        else:
            pos, bigestLossNumber = locateBigOne(LnewLoss)
            #print("pos", pos)
            # print("bigestLossNumber", bigestLossNumber)
            # if bigestLossNumber > loss or (bigestLossNumber == loss and (SentenceNumLengthCount(sentencelist, Lnew[pos]) > SentenceNumLengthCount(sentencelist, newCandi))):
            if bigestLossNumber > loss:
                # print("replacing", pos)
                LnewLoss[pos] = loss
                Lnew[pos] = newCandi.copy()
                Nvec[pos] = ReconVec.copy()
    else:
        if (len(Chosen) < beamSize):
            Chosen.append(newCandi.copy())
            ChosenLoss.append(loss)

        else:
            pos, bigestLossNumber = locateBigOne(ChosenLoss)
            # if bigestLossNumber > loss or (bigestLossNumber == loss and (SentenceNumLengthCount(sentencelist, Chosen[pos]) > SentenceNumLengthCount(sentencelist, newCandi))):
            if bigestLossNumber > loss:
                ChosenLoss[pos] = loss
                Chosen[pos] = newCandi.copy()
    return Lnew, LnewLoss, Chosen, ChosenLoss, Nvec

def extendable(sentenceList, imptList, TargetSenNum):
    if (len(imptList)+1 < TargetSenNum):
        return True
    else:
        return False

def locateBigOne(List):
    #print("List", List)
    bigestNumber = 0
    bigestPos = 0
    listLength = len(List)
    #print("Len", listLength)
    for i in range(listLength):
        if List[i] > bigestNumber:
            bigestNumber = List[i]
            bigestPos = i
    return bigestPos, bigestNumber

def locateSamllestOne(List):
    smallestNumber = 100
    smallestPos = 0
    listLength = len(List)
    for i in range(listLength):
        if List[i] < smallestNumber:
            smallestNumber = List[i]
            smallestPos = i
    return smallestPos, smallestNumber

def AverageDvecGenProcess(intcom, intsvm, sennum, Encoder, MODELD):
    Svecs = []
    targetvecs = []
    SvecList = []
    for senk in range(sennum):
        input_tensor = torch.tensor(intcom[senk].copy(), dtype=torch.long, device=device)
        input_tensor = input_tensor.to(device)
        Encoder = Encoder.to(device)
        encodeHidden = ProcSgen(Encoder, input_tensor)

        CombinedVec = combineForthAndBack(encodeHidden)

        Svecs.append(CombinedVec.clone().detach())
        targetvecs.append(CombinedVec.clone().detach())
        #SvecList.append(CombinedVec.clone().detach().cpu().numpy().tolist().copy())

    #print("length", len(intcom), len(intsvm))
    #print(intsvm)

    SVMvec = torch.tensor(intsvm, dtype=torch.float, device=device)
    #SVMvec = torch.softmax(SVMvec, 0)
    SVMvec = SVMvec.view(-1, 1).repeat(1, 2*MODELD)
    SVMvec = SVMvec.to(device)

    Svecstensor = torch.cat(Svecs)

    #print("Svecs", Svecs[0])

    #print("Size", Svecstensor.size())
    #print("Before Svecstensor", Svecstensor)


    Svecs_SVM = torch.mul(Svecstensor, SVMvec)
    Svecs_SVM = Svecs_SVM.detach()

    for i in Svecs_SVM:
        SvecList.append(i.clone().detach().cpu().numpy().tolist().copy())

    Svecs_SVM = Svecs_SVM.to(device)

    #print("Svecs_SVM", Svecs_SVM)

    return Svecs_SVM, SvecList


def ProcSgen(encoder, input_tensor):#inputtensor是一句话
    input_length = input_tensor.size(0)
    encoder_hidden = encoder.initHidden()
    encoder_hidden = encoder_hidden.to(device)
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
    #return encoder_hidden[0].clone().detach()#Bidirection hidden, only take forward
    return encoder_hidden.clone().detach()  # Bidirection hidden, take them both

def combineForthAndBack(encodeHidden):
    forthVec = encodeHidden[0]#[1,400]
    backVec = encodeHidden[1]#[1,400]
    answer = torch.cat((forthVec, backVec), 1)#[1,800]
    #print("answer", answer.size())
    return answer


if __name__ == "__main__":

    # # SDS 데이터셋의 경우
    #BugSum("ADS", b, 1) # beamsize 14 fscore:0.433
    #BugSum("ADS", b, 0) # beamsize 10 fscore:0.456

    beam_size = [8,9,10,11,12,13,14,15] # 9가 제일 좋음 text ranking * 100
    # for b in beam_size:
    #     BugSum("SDS", b, 0.5)

    #BugSum("SDS", 14, 1) # fscore 0.368 (ignore list 무시하고 모든 문장을 고려하여 요약문 생성)
    #
    alpha = [0, 0.25, 0.5, 0.75]
    # alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    # beta = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    # gamma = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    a = 1
    b = 0
    g = 0
    weight = [10, 100, 1000, 10000]
    #flist = []
    # ADS의 경우: 10000, SDS의 경우: 100
    w = 10000
    sumlen = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    #sumlen = [0.15, 0.25, 0.35, 0.45]
    # beam size SDS : 8 ADS : 8
    beamSize = 8
    l = 0.3

    preli = []
    recli = []
    f1li = []
    # #print("Beam Size:", beamSize)
    # for b in alpha:
    #     for b in beta:
    #         for g in gamma:
    #             if (a + b + g) != 1:
    #                 continue

        # g = 1 - b
    acc, recall, f1score = BugSum("ADS", beamSize, a, b, g, w, l)
    preli.append(acc)
    recli.append(recall)
    f1li.append(f1score)
    print("=======================================")
    print("sentence weight:", w)
    print("=======================================")
    print("alpha: ", a, "beta: ", b, "gamma: ", g)
    print("=======================================")
    #print("len", l, "%")
    print("avgPrecision:",acc)
    print("avgRecall:", recall)
    print("avgFscore:", f1score)
            #flist.append(f1score)

    # print("=======================================")
    # print("avgPrecision:", max(preli))
    # print("avgRecall:", max(recli))
    # print("avgFscore:", max(f1li))


        # we = str(w)
        # plt.plot(a, flist, marker='o', label=we)
        # flist = []

    # plt.xticks(a, labels=['0', '0.25', '0.5', '0.75', '1'])
    # plt.legend()
    # plt.show()
    #BugSum("ADS", 9, 0)

