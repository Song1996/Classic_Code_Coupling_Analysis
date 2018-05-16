from collections import defaultdict
from time import time
import numpy as np
from itertools import combinations
import sys
from sklearn.metrics import roc_auc_score

PROJECT_NAME = 'git'
START = 0
POS = 3000
END = 10000

class TARMAQ:
    def __init__(self, train_data, file2num):
        self.H = train_data
        self.H = list(map(lambda x:frozenset(x), self.H))
        self.file2num = file2num
        self.num_files = len(file2num)

    def predict(self, Qlist,k1,k2):
        H = self.H
        Qlist = map(lambda x:frozenset(x), Qlist)
        Flist = []
        Plist = []
        for Q in Qlist:
            H_f = set({})
            k = 0
            for T in H:
                if len(Q.intersection(T)) == k:
                    H_f = H_f.union(set([T]))
                elif len(Q.intersection(T)) > k:
                    k = len(Q.intersection(T))
                    H_f = set([T])
            R = defaultdict(lambda :[0.0, 0.0])
            for T in H_f:
                Q_ = Q.intersection(T)
                kappa_down = 0.0
                for x_ in H:
                    if x_.issuperset(Q_):
                        kappa_down += 1
                for x in T - Q_: 
                    Q_ux = Q_.union(set([x]))
                    phi = 0.0
                    for x_ in H:
                        if x_.issuperset(Q_ux):
                            phi += 1
                    R[(Q_, x)][0] = phi/(len(H)+1e-10)
                    R[(Q_, x)][1] = phi/(kappa_down+1e-10)
            if len(R) == 0:
                Flist.append([])
                Plist.append({})
                continue
            (maxR_0,minR_0) = (max(map(lambda k:R[k][0],R.keys())), min(map(lambda k:R[k][0],R.keys())))
            (maxR_1,minR_1) = (max(map(lambda k:R[k][1],R.keys())), min(map(lambda k:R[k][1],R.keys())))
            R_transed = defaultdict(lambda:[0,0])
            for k in R:
                R_transed[k][0] = (R[k][0] - minR_0 + 1e-10)/(maxR_0 - minR_0 + 1e-10)
                R_transed[k][1] = (R[k][1] - minR_1 + 1e-10)/(maxR_1 - minR_1 + 1e-10)
            R_s = sorted(R_transed.keys(), key = lambda x:R_transed[x][0]*k1 + R_transed[x][1]*k2, reverse = True)
            F = [x[1] for x in R_s]
            P = {x[1]:R_transed[x][0]*k1 + R_transed[x][1]*k2 for x in R_s}
            Flist.append(F[:10])
            Plist.append(P)
        return Flist,Plist
        #return (R,R_transed)

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



def validation(test_data, tarmaq, K1,K2, train_file_set):
    print('K1, %f K2, %f'%(K1,K2))
    seen_navy = np.zeros(6)
    seen_prev = np.zeros(6)
    unseen_navy = np.zeros(6)
    unseen_prev = np.zeros(6) 
    seen_flag = False
    cnt = 0
    num_seen = 0
    num_unseen = 0
    for commit in test_data:
        if set(commit).issubset(train_file_set):
            seen_flag = True
            num_seen += 1
        else:
            num_unseen += 1	        
        cnt += 1
        if cnt%10 == 0:
            print(cnt)
        hit_condition = np.zeros(6)
        commit = frozenset(commit)
        q_cnt, hit_cnt = 0, np.zeros(6)
        for q in commit:
            q = frozenset({q})
            q_cnt += 1
            label = commit - q
            y,p = tarmaq.predict([q],K1, K2)
            y = y[0]
            p = p[0]
            if len(label.intersection(frozenset(y[:10]))) > 0:
                hit_cnt[0] += 1
            if len(label.intersection(frozenset(y[:5])))>0:
                hit_cnt[1] += 1
            if len(label.intersection(frozenset(y[:3])))>0:
                hit_cnt[2] += 1
            if len(label.intersection(frozenset(y[:1])))>0:
                hit_cnt[3] += 1            
            file_fortest = set(label).union(set(p.keys()))
            file_l = [1 if x in label else 0 for x in file_fortest]
            file_p = [p[x] if x in p else 0 for x in file_fortest]
            auc_score = 0.5
            if len(file_fortest) == len(label):
                auc_score = 0.5
            else:
                auc_score = roc_auc_score(file_l, file_p)
            hit_cnt[4] += auc_score
        if seen_flag:
            seen_navy += (hit_cnt/q_cnt)
        else:
            unseen_navy += (hit_cnt/q_cnt)
        if len(commit) == 2:
            if seen_flag:
                seen_prev += (hit_cnt/q_cnt)
            else:
                unseen_prev += (hit_cnt/q_cnt)
            continue         
        
        q_cnt, hit_cnt = 0, np.zeros(6)
        for q in commit:
            label = frozenset({q})
            q_cnt += 1
            q = commit - label
            y,p = tarmaq.predict([q],K1, K2)
            y = y[0]
            p = p[0]
            if len(label.intersection(frozenset(y[:10]))) > 0:
                hit_cnt[0] += 1
            if len(label.intersection(frozenset(y[:5])))>0:
                hit_cnt[1] += 1
            if len(label.intersection(frozenset(y[:3])))>0:
                hit_cnt[2] += 1
            if len(label.intersection(frozenset(y[:1])))>0:
                hit_cnt[3] += 1        
            file_fortest = set(label).union(set(p.keys()))
            file_l = [1 if x in label else 0 for x in file_fortest ]
            file_p = [p[x]  if x in p else 0 for x in file_fortest]
            auc_score = 0.5
            if len(file_fortest) == len(label):
                auc_score = 0.5
            else:
                auc_score = roc_auc_score(file_l, file_p)
            hit_cnt[4] += auc_score
        if seen_flag:
            seen_prev += (hit_cnt/q_cnt)
        else:
            unseen_prev += (hit_cnt/q_cnt)    
    print(seen_navy)
    print(unseen_navy)
    print(seen_prev)
    print(unseen_prev)
    print((seen_navy*num_seen + unseen_navy*num_unseen)/(num_seen + num_unseen))
    print((seen_prev*num_seen + unseen_prev*num_unseen)/(num_seen + num_unseen))
    print(num_seen)
    print(num_unseen)
    return [seen_navy, unseen_navy, seen_prev, unseen_prev]

hit_dict = dict({})
for PROJECT_NAME in [sys.argv[1]]:  #['git','subversion','wine','rails']:
    print('project', PROJECT_NAME)
    train_data, test_data, file2num = load_data(PROJECT_NAME,START,POS,END)
    tarmaq = TARMAQ(train_data, file2num)
    train_file_set = set({})
    for c in train_data:
        train_file_set.update(set(c))
    hit_dict[PROJECT_NAME]=validation(test_data, tarmaq, 1,1, train_file_set)

