import sys,os


project = sys.argv[1]
print('select the newest 10000 commits from '+project)

f = open('projects/file_list/'+project+'_file_list.txt','r')
lines = ''.join(f.readlines())
f.close()

commits = [x.split('\n') for x in lines.split('\n\n')]
commits = list(filter(lambda x:len(x)>2 and len(x) < 5,commits))

f = open('projects/selected_file_list/'+project+'_selected_file_list.txt','w')
f.write( '\n'.join(','.join(commit) for commit in commits[:10000]) )
f.close()

