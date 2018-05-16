import os
import random
import numpy as np
import torch
from time import time
import itertools

class Data_gener:
    def sentence_split(self, file_name):
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        file_name = ''.join(map(lambda char: char if char not in filters else ' ', file_name))
        file_name = file_name.split(' ')
        return file_name[:self.max_sequence_len]
    
    def fit_file_name(self, file_name_list, filter_id_list = []):
        cnt = 0
        for ix in range(len(file_name_list)):
            file_name = file_name_list[ix]

            if file_name not in self.file_id_dict:
                self.file_id_dict[file_name] = len(self.file_id_dict)
                file_name_words = self.sentence_split(file_name)
                for word in file_name_words:
                    if word not in self.word_dict:
                        self.word_dict[word] = len(self.word_dict)
                file_name_words = list(map(lambda x:self.word_dict[x], file_name_words))
                self.file_name_dict[self.file_id_dict[file_name]] = file_name_words
            file_id = self.file_id_dict[file_name]

            if file_id not in filter_id_list:
                file_name_list[cnt] = file_id
                cnt += 1
        return file_name_list[:cnt]


    def __init__(self, project , max_sequence_len =20, batch_size = 512):
        #random.seed(1)
        self.project = project
        self.max_sequence_len = max_sequence_len
        self.batch_size = batch_size

        self.file_id_dict = dict({})
        self.word_dict = dict({' ':0})
        self.file_name_dict = dict({})
        self.commit_dict = dict({})
        self.word_reverse_dict = dict({})

        print('project: ', project)
        print('max_sequence_len: ', max_sequence_len)
        print('batch_size: ', batch_size)
        self.easy_init()

    def first_init(self):
        commits = [ line.strip('\n').split(',') for line in open('projects/selected_file_list/%s_selected_file_list.txt'%self.project,'r') ]
        assert len(commits) == 10000, "commits number is not 10000"
        
        print('cd /home/song/change_recommend_pytorch/projects/%s'%self.project)
        print(os.popen('cd /home/song/change_recommend_pytorch/projects/%s'%self.project).read())
        print(os.popen('pwd').read())
        print('hi')
        l = len(os.popen('cd /home/song/change_recommend_pytorch/projects/%s && git reset --hard 3f281a3baad9f5f8f875da902718a1d5d3dc0d9f' %self.project).read())
        print('hi')
        for commit_id in range(len(commits)):
            commit = commits[commit_id]
            commit_hash = commit[0]
            commit_files = self.fit_file_name(commit[1:])
            l = len(os.popen('cd /home/song/change_recommend_pytorch/projects/%s && git checkout %s &> /dev/null'%(self.project, commit_hash)).read())
            all_files = os.popen('cd /home/song/change_recommend_pytorch/projects/%s && find . -type f'%(self.project)).read().split('\n')
            all_files = list(map(lambda x:x[2:], filter(lambda x:True if len(x)>2 else False, all_files)))
            all_files = self.fit_file_name(all_files, filter_id_list = commit_files)
            commits[commit_id] = [commit_hash, commit_files, all_files]
            self.commit_dict[commit_id] = [commit_files, all_files]

            for k in self.file_name_dict.keys():
                file_name = self.file_name_dict[k]
                file_name = np.lib.pad(file_name, [0, self.max_sequence_len - len(file_name)], 'constant')
                self.file_name_dict[k] = file_name

            commits = [self.commit_dict[ix] for ix in range(10000)]
            self.train_commits = [self.commit_dict[ix] for ix in range(3000,10000)]
            self.test_commits = [self.commit_dict[ix] for ix in range(3000)]
            self.save_dict()
            


    
    def easy_init(self):
        f_file_id_dict = open('/home/song/change_recommend_pytorch/projects/dicts/%s_file_id_dict.txt'%self.project, 'r')
        for line in f_file_id_dict:
            line = line.strip('\n').split(',')
            self.file_id_dict[line[0]] = int(line[1])
        f_file_id_dict.close()
        f_word_dict = open('/home/song/change_recommend_pytorch/projects/dicts/%s_word_dict.txt'%self.project, 'r')
        for line in f_word_dict:
            line = line.strip('\n').split(',')
            self.word_dict[line[0]] = int(line[1])
        f_word_dict.close()
        f_file_name_dict = open('/home/song/change_recommend_pytorch/projects/dicts/%s_file_name_dict.txt'%self.project, 'r')
        for line in f_file_name_dict:
            line = list(map(int,line.strip('\n').split(',')))
            assert len(line) == 21, "file_name_dict load error" 
            self.file_name_dict[line[0]] = np.array(line[1:])
        f_file_name_dict.close()
        f_commit_dict = open('/home/song/change_recommend_pytorch/projects/dicts/%s_commit_dict.txt'%self.project, 'r')
        for line in f_commit_dict:
            commit_id, commit_files, other_files = line.strip('\n').split(';')
            commit_id, commit_files, other_files = int(commit_id), list(map(int, commit_files.split(','))), list(map(int, other_files.split(',')))
            self.commit_dict[commit_id] = [commit_files, other_files]
        f_commit_dict.close()

        for k in self.word_dict.keys():
            self.word_reverse_dict[self.word_dict[k]] = k

        commits = [self.commit_dict[ix] for ix in range(10000)]
        self.train_commits = [self.commit_dict[ix] for ix in range(3000,10000)]
        self.test_commits = [self.commit_dict[ix] for ix in range(3000)]

    def save_dict(self):
        f_file_id_dict = open('/home/song/change_recommend_pytorch/projects/dicts/%s_file_id_dict.txt'%self.project, 'w')
        for k in self.file_id_dict.keys():
            f_file_id_dict.write(','.join([str(k), str(self.file_id_dict[k]) ]) + '\n')
        f_file_id_dict.close()
        f_word_dict = open('/home/song/change_recommend_pytorch/projects/dicts/%s_word_dict.txt'%self.project, 'w')
        for k in self.word_dict.keys():
            f_word_dict.write(','.join([str(k), str(self.word_dict[k]) ]) + '\n')
        f_word_dict.close()
        f_file_name_dict = open('/home/song/change_recommend_pytorch/projects/dicts/%s_file_name_dict.txt'%self.project, 'w')
        for k in self.file_name_dict.keys():
            f_file_name_dict.write(','.join([str(k)]+ list(map(str,self.file_name_dict[k]))) + '\n')
        f_file_name_dict.close()
        f_commit_dict = open('/home/song/change_recommend_pytorch/projects/dicts/%s_commit_dict.txt'%self.project, 'w')
        for k in self.commit_dict.keys():
            f_commit_dict.write(';'.join([str(k), ','.join(map(str,self.commit_dict[k][0])), ','.join(map(str,self.commit_dict[k][1]))]) + '\n')
        f_commit_dict.close()

    def gener(self, train_or_test, torch_or_numpy = 'torch', augmentation = False):
        project = self.project
        max_sequence_len = self.max_sequence_len
        batch_size = self.batch_size
        
        if train_or_test == 'train':
            commits = self.train_commits
        elif train_or_test == 'test':
            commits = self.test_commits
        else:
            assert True, "train_or_test input error"
        if augmentation:
            for commit_ix in range(len(commits)):
                commit_files = commits[commit_ix][0]
                aug_files = list(itertools.permutations(commit_files,2))
                for aug_id in aug_files:
                    if aug_id not in self.file_name_dict.keys():
                        names = list(map(lambda x:list(self.file_name_dict[x]), aug_id))
                        names = list(map(lambda x: x[:x.index(0)] if 0 in x else x, names))
                        aug_name = sum(names,[])[:self.max_sequence_len]
                        aug_name = np.lib.pad(aug_name, [0, self.max_sequence_len - len(aug_name)], 'constant')
                        self.file_name_dict[aug_id] = aug_name
                commits[commit_ix][0] = commits[commit_ix][0] + aug_files
            
        while True:
            
            num_commits   = len(commits)
            left_samples  = np.zeros([batch_size, max_sequence_len], dtype=np.int64)
            right_samples = np.zeros([batch_size, max_sequence_len], dtype=np.int64)
            label_samples = [0]* int(batch_size)

            shuffled_ix = list(range(batch_size))
            random.shuffle(shuffled_ix)

            for sample_ix in shuffled_ix[:int(batch_size/2)]:
                sample_commit = commits[random.randint(0,num_commits-1)][0]
                left_samples[sample_ix], right_samples[sample_ix]  = map(lambda x:self.file_name_dict[x], random.sample(sample_commit,2))
                label_samples[sample_ix] = [1]
            
            for sample_ix in shuffled_ix[int(batch_size/2):]:
                sample_commit_files, sample_other_files = commits[random.sample(range(0,num_commits),1)[0]]
                left_samples[sample_ix] = self.file_name_dict[random.sample(sample_commit_files,1)[0]]
                right_samples[sample_ix] = self.file_name_dict[random.sample(sample_other_files,1)[0]]
                label_samples[sample_ix] = [0]

            if torch_or_numpy == 'torch':
                yield [torch.LongTensor(left_samples), torch.LongTensor(right_samples), torch.Tensor(label_samples)]
            elif torch_or_numpy == 'numpy':
                yield ([left_samples, right_samples], np.array(label_samples))

    def commit_validation_generation(self, commit_ix, file_ix):
        commit = self.test_commits[commit_ix%len(self.test_commits)]
        assert file_ix in range(len(commit[0])), "commit_validation_gener error, file_ix not in commit"
        right_files_id = commit[0][:file_ix] + commit[0][file_ix+1:] + commit[1]
        num_other_files = len(right_files_id)
        label_samples = torch.Tensor([1]*(len(commit[0])-1) + [0]*len(commit[1]))
        left_samples = torch.LongTensor(np.array([self.file_name_dict[commit[0][file_ix]]]*num_other_files))
        right_samples = torch.LongTensor( np.array(list(map(lambda x:self.file_name_dict[x], right_files_id))) )
        return left_samples, right_samples, label_samples

    def reverse_translate(self,X_array):
        if type(X_array) == torch.autograd.variable.Variable:
            X_array = X_array.cpu().data.numpy()
        elif type(X_array) in [torch.FloatTensor, torch.cuda.FloatTensor, torch.cuda.LongTensor, torch.LongTensor]:
            X_array = X_array.cpu().numpy()
        sentences_list = ['' for ix in range(X_array.shape[0])]
        for ix in range(X_array.shape[0]):
            digital_sentence = list(X_array[ix,:])
            sentence = list(map(lambda x:self.word_reverse_dict[x], digital_sentence))
            sentences_list[ix] = '/'.join(filter(lambda x:False if x == ' ' else True, sentence))
        return sentences_list


