import os
import pandas as pd
from Util import writecsv, DataListListProcess
def SentenceDatainput(filename):
    sentencenumberlist = []
    sentencelist = []
    commentauthor = []
    sentenceInvolved = []
    imptSenInvolveList = []
    title = []
    imptsentence = ""

    readed_data = open(filename, encoding='utf-8')
    lines = readed_data.read().split('\n')

    linenum = len(lines)

    reportNum = 0


    for i in range(linenum):
        # if ('<BugReport ID = "' in lines[i]):
        #     reportNum += 1
        if ('<Title>' in lines[i]):
            t_start = lines[i].index('-') + 1
            t_end = t_start + lines[i][t_start:].index('"')
            title_sentence = lines[i][t_start:t_end]
            title_sentence = title_sentence.strip()

            title.append(title_sentence)


        if (lines[i][:6] == "<From>"):
            if (len(commentauthor)!=0):
                sentenceInvolved.append(imptSenInvolveList.copy())
                imptSenInvolveList.clear()
            lines[i] = lines[i][6:-7]
            #lines[i] = lines[i].replace('\'', ' ').replace('_', ' ').replace('.', ' ').replace('-', ' ')
            lines[i] = ' '.join(lines[i].split())

            #commentauthor.append(lines[i][11:])
            commentauthor.append(lines[i][1:-1])
            continue

        # Sentence Number 추출
        if (sentence_num_judgement(lines[i])):
            if (len(sentencenumberlist) != 0):
                #sentencelist.append(imptsentence)
                imptsentence = ""

            start = lines[i].index('"') + 1
            end = start + lines[i][start:].index('"')
            sentencenum = lines[i][start:end]

            sentencenumberlist.append('\''+sentencenum)

            # if sentencenum.endswith('>'):
            #     sentencenum = sentencenum[:-1]


            sentence = lines[i][end+2:-11]
            sentence = sentence.strip()

            sentence = sentence.replace("&amp;gt;", '>').replace("&gt;", '>')
            sentencelist.append(sentence)

            #print(sentence)

            # imptSenInvolveList.append(lines[i])
            imptSenInvolveList.append(len(sentencenumberlist) - 1)

        else:
            imptsentence = imptsentence + lines[i]

    #sentencelist.append(imptsentence)
    sentenceInvolved.append(imptSenInvolveList)

    return sentencenumberlist, sentencelist, commentauthor, sentenceInvolved, title

def readFolderFile(FolderName):
    newfileList = []
    for _, _, files in os.walk(FolderName):
        for file in files:
            newfileList.append(FolderName + "\\" + file)
    return newfileList

def sentence_num_judgement(astring):

    if "Sentence ID" in astring:
        #print(astring)
        return True
    else:
        return False


def DataIn(datasetName):
    #fileNameList = readFolderFile("Sentence\\" + datasetName)
    fileNameList = readFolderFile("Dataset\\" + datasetName)
    writer = pd.ExcelWriter("BugSum_Data_"+datasetName+".xlsx", engine='xlsxwriter')
    data = {}
    counter = 0
    #print(fileNameList)
    for fileName in fileNameList:
        print("fileName", fileName)
        fileNumber = fileName[12:14]
        print(fileNumber)
        sentencenumberlist, sentencelist, commentauthor, sentenceInvolved, title = SentenceDatainput(fileName)
        #print("sentencenumberlist", sentencenumberlist)
        #print("sentencelist", sentencelist)
        #print("commentauthor", commentauthor)
        #print("sentenceInvolved", sentenceInvolved)
        print("title", title)

        data["Sentence"] = sentencelist
        data["SenNumber"] = sentencenumberlist
        data["CommentAuthor"] = commentauthor
        data["ComSenNum"] = DataListListProcess(sentenceInvolved)
        data["Title"] = title

        writecsv(writer, fileNumber, data)
        counter = counter + 1
        #print("sheet", fileName)


    writer.save()

    writer.close()
if __name__ == "__main__":
    #DataIn("ADS")

    # ADS 에서 조건에 맞는 일부 데이터 셋
    #DataIn("EST")
    DataIn("ADS")

    # TES + EST 합친 것것
    #DtaIn("XDS")

    #print(readFolderFile("Dataset"))
