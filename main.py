import numpy as np
from itertools import combinations

PROJECT_NAME = 'rails'
START = 0
POS = 3000
END = 10000
K1_list = [0.1,0.2,0.3,0.4,0.5]
K2_list = [3,5,10,15,20]

def load_data(project_name,start,pos,end):
    lines = open(project_name+'_selected_file_list.txt').readlines()
    file2num = dict({})
    num2file = dict({})
    data = []
    for line in lines[start:end]:
        line = line.strip('\n').split(',')[1:]
        for file_str in line:
            if file_str not in file2num:
                file2num[file_str] = len(file2num)
        data.append([file2num[file_str] for file_str in line])
    train_data = data[pos:end]
    test_data  = data[start:pos]
    print('num train commits', len(train_data))
    print('num test commits', len(test_data))
    print('num file str', len(file2num))
    return train_data, test_data, file2num

def predict(q, u, s, K1, K2):
    file_num = s.shape[0]
    clusters = []
    output_dict = dict({})
    for ix in range(int(file_num*K1)):
        cluster_flag = list(abs(u[:,ix]) > K2*np.mean(abs(u[:,ix])))
        cluster = [ix for ix in range(file_num) if cluster_flag[ix]]
        clusters.append(frozenset(cluster))
    for cluster in clusters:
        if q.issubset(cluster):
            for ix in cluster-q:
                if ix not in output_dict:
                    output_dict[ix] = 0
                output_dict[ix] += 1
    return sorted(output_dict.keys(), key = lambda x:output_dict[x], reverse = True)

def validation(test_data, u, s, K1,K2):
    print('K1, %f K2, %f'%(K1,K2))
    hit = np.zeros([4,4])
    for l in range(1,5):
        hit_condition = np.zeros(4)
        for commit in test_data:
            if len(commit) < l:
                continue
            gen_q = combinations(commit, l)
            commit = frozenset(commit)
            q_cnt, hit_cnt = 0, np.zeros(4)
            for q in gen_q:
                q = frozenset(q)
                q_cnt += 1
                label = commit - q
                y = predict(q, u, s, K1, K2)
                if label.intersection(frozenset(y[:10])):
                    hit_cnt[0] += 1
                if label.intersection(frozenset(y[:5])):
                    hit_cnt[1] += 1
                if label.intersection(frozenset(y[:3])):
                    hit_cnt[2] += 1
                if label.intersection(frozenset(y[:1])):
                    hit_cnt[3] += 1
            hit_condition += (hit_cnt/q_cnt)
        #print(hit_condition/len(test_data))
        hit[l-1,:] = hit_condition/len(test_data)
    print(hit)
    return hit
        
for PROJECT_NAME in ['git','subversion','wine','rails']:
    print('project', PROJECT_NAME)
    train_data, test_data, file2num = load_data(PROJECT_NAME,START,POS,END)
    file_num = len(file2num)
    comatrix = np.zeros([file_num, file_num])
    for commit in train_data:
        for i,j in combinations(commit,2):
            comatrix[i,j] += 1
            comatrix[j,i] += 1
    u,s,v = np.linalg.svd(comatrix)
    print('begin validation')
    result_dict = dict({})
    for K1 in K1_list:
        for K2 in K2_list:
            hit = validation(test_data, u, s, K1, K2)
            result_dict[(K1,K2)] = hit
    f = open(PROJECT_NAME+'_svd_result.txt','w')
    for k in result_dict:
        hit = result_dict[k]
        f.write('%f,%f,'%(k[0],k[1]))
        f.write(','.join(list(map(str,hit[0]))+list(map(str,hit[1]))+list(map(str,hit[2]))+list(map(str,hit[3])))+'\n')
    f.close()
    best_result = np.zeros([4,4])
    for i in range(4):
        for j in range(4):
            best_result[i,j] = max(map(lambda k:result_dict[k][i,j], result_dict.keys()))
    print('best result:')
    print(best_result)
