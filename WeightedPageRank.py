# request, bs4, pprint, summa 모듈 설치해야함

import requests
from bs4 import BeautifulSoup
from pprint import pprint as p
import os.path
from Util import readcsv, SheetName, stopwordinput, preprocess_spacy

# 3종류 연관관계의 weight
w_dup = 1
w_dep = 1
w_blo = 1
# 버그 리포트 그래프 형성을 위한 탐색 깊이
depth = 1
# 중복 관계(duplicates)를 사용할 것인지 여부
dup = True
# 의존 관계(dependson, blocks)를 사용할 것인지 여부
dep_blo = True

# 연관 관계가 많은 버그 리포트들에 대해 최대 N개의 연관 관계만 이용하도록 제한 (추가 실험용)
max_rel = 3

# 요약 대상 버그 리포트
num = 1439860

# 코퍼스 구축을 위한 flag
get_answer = False
get_text = False

# 공용 추가 stopwords
additional_stopwords = ['bug']
# 개별 버그 리포트의 추가 stopwords
from test import stopwords

if num in stopwords:
    additional_stopwords += stopwords[num]


# 버그 리포트 크롤링 함수
def get(num, t=False):
    # html 파일을 캐싱한 경우 로컬 파일 읽기
    if os.path.isfile('html/' + str(num) + '.html'):
        with open('html/' + str(num) + '.html', 'r') as f:
            html_text = f.read()
    # html 파일을 캐싱하지 않은 경우 웹에서 읽고 html 파일 저장
    else:
        html_text = requests.get('https://bugzilla.mozilla.org/show_bug.cgi?id=' + str(num)).text
        with open('html/' + str(num) + '.html', 'w') as f:
            f.write(html_text)

    # 크롤링한 결과 html 객체
    content = BeautifulSoup(html_text, 'html.parser')

    # 해당 버그 리포트에 걸려있는 중복(duplicates) 관계 버그 리포트들 수집
    duplicates = []
    temp = content.find('a', href='https://wiki.mozilla.org/BMO/UserGuide/BugFields#duplicates')
    if temp:
        counter = 0
        for a in temp.parent.next_sibling.next_sibling.find_all('a'):
            counter += 1
            if counter == max_rel:
                continue
            if a.text.isdigit():
                duplicates.append(int(a.text))

    # 해당 버그 리포트에 걸려있는 의존(dependson) 관계 버그 리포트들 수집
    dependson = []
    temp = content.find('a', href='https://wiki.mozilla.org/BMO/UserGuide/BugFields#dependson')
    if temp:
        counter = 0
        for a in temp.parent.next_sibling.next_sibling.find_all('a'):
            counter += 1
            if counter == max_rel:
                continue
            if a.text.isdigit():
                dependson.append(int(a.text))

    # 해당 버그 리포트에 걸려있는 역의존(blocks) 관계 버그 리포트들 수집
    blocks = []
    temp = content.find('a', href='https://wiki.mozilla.org/BMO/UserGuide/BugFields#blocks')
    if temp:
        counter = 0
        for a in temp.parent.next_sibling.next_sibling.find_all('a'):
            counter += 1
            if counter == max_rel:
                continue
            if a.text.isdigit():
                blocks.append(int(a.text))

    # 해당 버그 리포트의 문장들 수집
    temp = content.find_all('div', class_='change-set')
    comments = []
    for comment in temp:
        if comment.pre:
            comments.append(comment.pre.text.replace('\n', ' '))
        else:
            x = comment.find_all(['p', 'li'])
            temp_str = ''
            for y in x:
                temp_str += '\n' + y.text
            comments.append(temp_str)

    # 버그 리포트들의 타입, 연관 관계 개수 등을 분석하기 위한 추가 실험 전용 코드
    if t == True:
        import re
        x = content.find('h1', text=re.compile('Access Denied'))
        if x:
            return None, None, None, None, 'access'
        x = content.find('h1', text=re.compile('Missing Bug ID'))
        if x:
            return None, None, None, None, 'access'
        temp = content.find('span', id='field-value-bug_type')
        br_type_raw = temp.span.text
        if 'enhancement' in br_type_raw:
            br_type = 'e'
        elif 'defect' in br_type_raw:
            br_type = 'd'
        elif 'task' in br_type_raw:
            br_type = 't'
        else:
            exit()

        return duplicates, dependson, blocks, comments, br_type

    # get 함수의 리턴값
    # duplicates : 해당 버그 리포트의 중복 관계 버그 리포트들의 번호의 list
    # dependson : 해당 버그 리포트의 의존 관계 버그 리포트들의 번호의 list
    # blocks : 해당 버그 리포트의 역의존 관계 버그 리포트들의 번호의 list
    return duplicates, dependson, blocks, comments


# 인풋으로 요약 대상 버그 리포트, 중복 관계 사용 여부, 의존 관계 사용 여부, 탐색 깊이를 받아
# 해당 버그 리포트를 root node로 하는 버그 리포트 그래프 형성
def get_all(num, dup=dup, dep_blo=dep_blo, depth=depth):
    # 버그 리포트 그래프의 노드들 (버그 리포트 넘버의 리스트)
    nodes = []
    # 버그 리포트 그래프의 노드들의 상세정보를 담은 리스트 (nodes와 인덱스를 공유)
    nodes_details = []
    # DFS를 위한 아직 방문하지 않은 노드들의 리스트
    yet = [num]

    # if False:
    #     while len(yet) != 0:
    #         print(yet)
    #         n = yet.pop()
    #         if not n in nodes:
    #             nodes.append(n)
    #             detail = list(get(n))
    #             nodes_details.append(detail)
    #             if dup:
    #                 yet += detail[0]
    #             if dep_blo:
    #                 yet += detail[1] + detail[2]

    # DFS 서치
    for i in range(depth + 1):
        next_yet = []
        for n in yet:
            if not n in nodes:
                nodes.append(n)
                detail = list(get(n))
                nodes_details.append(detail)
                if dup:
                    next_yet += detail[0]
                if dep_blo:
                    next_yet += detail[1] + detail[2]
        yet = list(set(next_yet))

    # 수집한 모든 노드들의 연관 관계 정보를 이용해서 엣지를 찾아 nodes_details에 추가
    for node in nodes:
        new_dup_nodes = []
        for dup_node in nodes_details[nodes.index(node)][0]:
            if dup_node in nodes:
                new_dup_nodes.append(dup_node)
        nodes_details[nodes.index(node)][0] = new_dup_nodes
        new_dep_nodes = []
        for dep_node in nodes_details[nodes.index(node)][1]:
            if dep_node in nodes:
                new_dep_nodes.append(dep_node)
        nodes_details[nodes.index(node)][1] = new_dep_nodes
        new_blo_nodes = []
        for blo_node in nodes_details[nodes.index(node)][2]:
            if blo_node in nodes:
                new_blo_nodes.append(blo_node)
        nodes_details[nodes.index(node)][2] = new_blo_nodes

    # node_details 순회하며 엣지 연결
    edges = {}
    for node in nodes:
        edges[node] = {}

    # TODO dup and dep/blo dependency
    for node in edges:
        if dep_blo:
            for dep_node in nodes_details[nodes.index(node)][1]:
                edges[node][dep_node] = 'dep'
                edges[dep_node][node] = 'blo'
            for blo_node in nodes_details[nodes.index(node)][2]:
                edges[node][blo_node] = 'blo'
                edges[blo_node][node] = 'dep'

        if dup:
            dup_nodes = [node] + nodes_details[nodes.index(node)][0]
            for node1 in dup_nodes:
                for node2 in dup_nodes:
                    edges[node1][node2] = 'dup'

    # 버그 리포트 그래프 리턴
    return nodes, edges, nodes_details


# 이름은 main인데 main 함수는 아니고 1개의 버그 리포트에 대해 RBRS를 돌리는 함수
# 타겟 버그 리포트나 파라미터들은 코드 상단의 전역 변수를 교체해가며 사용
def main():
    # summa 라는 textrank 패키지를 사용
    from summa.preprocessing.textcleaner import clean_text_by_sentences as _clean_text_by_sentences
    from summa.commons import build_graph as _build_graph
    from math import log10
    from summa.commons import remove_unreachable_nodes as _remove_unreachable_nodes
    from summa.pagerank_weighted import pagerank_weighted_scipy as _pagerank

    # 해당 버그 리포트에 대한 그래프 생성
    nodes, edges, nodes_details = get_all(num, dup=dup, dep_blo=dep_blo, depth=depth)

    # 버그 리포트의 전처리
    # 버그 리포트의 문장들을 수집
    texts = []
    for detail in nodes_details:
        # 긴 문장이 wrapping되는 경우 \n이 붙어서 이를 제거
        texts.append('\n'.join(detail[3]))

    preprocessed_texts = []
    # 문장 분리, stopwords 제거, stemming 등 전처리 (summa 사용)
    for t in texts:
        preprocessed_texts.append(_clean_text_by_sentences(t, 'english', additional_stopwords))
    # 코퍼스 구축용 코드
    if get_text:
        print()
        # p(preprocessed_texts[0])
        for x in preprocessed_texts[0]:
            p(x.text)
            exit()
        # print()
        print()
    # 분리된 문장 수집
    all_sentences = []
    all_sentences_token = []
    for t in preprocessed_texts:
        for tt in t:
            all_sentences_token.append(tt.token)
            all_sentences.append(tt)
    # import json
    # printable = []
    # for x in all_sentences:
    #     printable.append(x.text)
    # with open('db/temp.json', 'w') as f:
    #     f.write(json.dumps(printable, indent=2, ensure_ascii=False))

    # 그래프 형성
    graph = _build_graph(all_sentences_token)

    # 두 문장 사이의 유사성을 구하기 위해 겹치는 단어 수집
    def _count_common_words(words_sentence_one, words_sentence_two):
        return len(set(words_sentence_one) & set(words_sentence_two))

    # 두 문장 사이의 유사성을 구하는 함수 (jaccard similarity)
    def _get_similarity(s1, s2):
        words_sentence_one = s1.split()
        words_sentence_two = s2.split()

        common_word_count = _count_common_words(words_sentence_one, words_sentence_two)

        log_s1 = log10(len(words_sentence_one))
        log_s2 = log10(len(words_sentence_two))

        if log_s1 + log_s2 == 0:
            return 0

        return common_word_count / (log_s1 + log_s2)

    # 특정 문장이 어느 버그 리포트의 문장인지 역추적하는 함수
    def find_num(s):
        for i in range(len(preprocessed_texts)):
            for ss in preprocessed_texts[i]:
                if s == ss.token:
                    return nodes[i]

    # 모든 엣지를 순회하며 문장과 문장이 서로 연결되는지, 연결된다면 weight가 얼마인지 계산
    for sentence_1 in graph.nodes():
        for sentence_2 in graph.nodes():
            edge = (sentence_1, sentence_2)
            if sentence_1 != sentence_2 and not graph.has_edge(edge):
                similarity = _get_similarity(sentence_1, sentence_2)
                if similarity != 0:
                    weight = 0
                    if find_num(sentence_1) == find_num(sentence_2):
                        graph.add_edge(edge, similarity)
                    elif find_num(sentence_2) in edges[find_num(sentence_1)]:
                        if edges[find_num(sentence_1)][find_num(sentence_2)] == 'dup':
                            weight = w_dup
                        elif edges[find_num(sentence_1)][find_num(sentence_2)] == 'dep':
                            weight = w_dep
                        elif edges[find_num(sentence_1)][find_num(sentence_2)] == 'blo':
                            weight = w_blo
                        graph.add_edge(edge, similarity * weight)

    # 도달 불가능한 노드들을 제거 (유사성이 존재하지 않아 엣지가 연결되지 않은 노드들)
    _remove_unreachable_nodes(graph)

    # pagerank 스코어 계산 및 할당 (summa 이용)
    pagerank_scores = _pagerank(graph)
    for sentence in all_sentences:
        if sentence.token in pagerank_scores:
            sentence.score = pagerank_scores[sentence.token]
        else:
            sentence.score = 0

    final_scores = list(map(lambda x: x.score, preprocessed_texts[0]))
    for i in range(len(final_scores)):
        final_scores[i] = [i, final_scores[i]]

    # 코퍼스 구축을 위한 코드
    if get_answer:
        for i in range(len(preprocessed_texts[0])):
            s = preprocessed_texts[0][i]
            # print(i, s.text, s.score)
            print(i, s.text)
            print()

    # print("rbrs score")
    # print(final_scores)
    # 상위 25%만큼 문장 추출 => selected
    sorted_final_scores = list(reversed(sorted(final_scores, key=lambda x: x[1])))
    selected = []
    for i in sorted_final_scores[:round(len(final_scores) * ratio)]:
        selected.append(i[0])
    selected = sorted(selected)

    # 테스트 데이터 로드 => answer
    from test import test
    answer = test[num]

    # selected, answer 겹치는 개수 구함
    count = 0
    for s in selected:
        if s in answer:
            count += 1
    # precision, recall, f1 score 계산 및 출력
    precision = count / len(selected[:-1]) * 100
    recall = count / len(answer[:-1]) * 100
    if precision + recall == 0:
        fscore = 0
    else:
        fscore = 2 * precision * recall / (precision + recall)
    # print('precision =', precision)
    # print('recall =', recall)
    # print('f-score =', fscore)
    print(count, '/', len(selected))

    return precision, recall, fscore


# main()
# exit()
if __name__ == '__main__':
    # brs = [
    # 1455326,1460748,1491498,1488720,1473893,
    # 1484270,1482147,1468460,1485810,1453294,
    # 1454094,1487327,1465667,1484275,1487227,
    # 1463501,1462633,1481319,1450275,1484521,
    # 1470114,1461815,1461736,1487150,1481171,
    # 1464840,1450847,1484947,1463754,1482875,
    # 1481171,
    # 1464840,1450847,1484947,1463754,1482875]
    # brs = [1439860,1529593,1505090,1504787,1475043,1458585]

    # xx = 0
    # dd = 0
    # bb = 0
    # for i in brs:
    #     x, d, b, _ = get(i)
    #     xx += 1 if len(x) > 0 else 0
    #     dd += 1 if len(d) > 0 else 0
    #     bb += 1 if len(b) > 0 else 0
    # print(xx, dd, bb)
    # exit()

    # 탐색 깊이 전역변수 세팅
    depth = 1
    # 버그 리포트들에 대한 개별 실험 결과값들
    ps_before = []  # prst precision
    rs_before = []  # prst recall
    fs_before = []  # prst f1 score
    ps_after = []  # rbrs precision
    rs_after = []  # rbrs recall
    fs_after = []  # rbrs f1 score
    brs = [1547317, 1545401, 1541089, 1533337, 1526017, 1520588, 1490195, 1483298, 1439860, 1529593, 1428331, 1396216,
           1195386, 1432858, 1381384, 1367146, 1534982, 1519087, 1505090, 1504787, 1490327, 1474757, 1543354, 1514809,
           1499000, 1486613, 1475831, 1475043, 1465555, 1458585, 1529593, 1529593, 1529593, 1529593]

    # 실험을 위한 대상 버그 리포트들
    # brs = [1439860,1529593,1505090,1504787,1475043,1458585]
    # brs = [1458585]

    # 각각 버그 리포트들에 대한 실험 결과값 계산 및 출력
    for br in brs:
        print('BR.' + str(br))
        num = br

        dep_blo = False
        print('PRST ', end='')
        p, r, f = main()
        ps_before.append(p)
        rs_before.append(r)
        fs_before.append(f)

        dep_blo = True
        print('RBRS ', end='')
        p, r, f = main()
        ps_after.append(p)
        rs_after.append(r)
        fs_after.append(f)
        print()
    print('===')
    print('result(pre,rec,f1)')
    # print(ps_before)
    # print(rs_before)
    # print(fs_before)
    # print(ps_after)
    # print(rs_after)
    # print(fs_after)
    # print('before')
    print('PRST(no BRC) :', round(sum(ps_before) / len(brs), 2), round(sum(rs_before) / len(brs), 2),
          round(sum(fs_before) / len(brs), 2))
    # print('after')
    print('RBRS(no BRC) :', round(sum(ps_after) / len(brs), 2), round(sum(rs_after) / len(brs), 2),
          round(sum(fs_after) / len(brs), 2))

    # print('asdf')
    # print(ps_before)
    # print(ps_after)
    exit()

    # for i in range(1450000, 1451000):
    #    _, _, _, _, t = get(i, t=True)
    #    print(i, t)
    # exit()

    # n_br = [0, 0, 0, 0] # d, e, t, access
    # n_dup = [0, 0, 0]
    # n_depblo = [0, 0, 0]

    # d = {'d': 0, 'e': 1, 't': 2}

    # for i in range(1455103, 1455104):
    #     dup, dep, blo, _, t = get(i, t=True)

    #     if t == 'access':
    #         n_br[3] += 1
    #     else:
    #         n_br[d[t]] += 1
    #         if len(dup) > 0:
    #             n_dup[d[t]] += 1
    #         if len(dep) + len(blo) > 0:
    #             n_depblo[d[t]] += 1

    #     print(sum(n_br), sum(n_dup), sum(n_depblo))
    # print()
    # print(n_br, n_dup, n_depblo)

    # ndep = 0
    # nblo = 0
    # for i in range(1450000, 1500000):
    #     print(i)
    #     _, dep, blo, _ = get(i)
    #     if len(dep) > 0:
    #         ndep += 1
    #     if len(blo) > 0:
    #         nblo += 1

    # print(dep, blo)



