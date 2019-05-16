import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


#Part-1 create the map from the output file
f = open("output-2.txt","r")
content = f.readlines()
scores={}#Indexed by tuple of (Observation_Count,Target_Count,Radar_Radius,Iteration_Number)
avg_scores={}#Indexed by tuple of (Observation_Count,Target_Count,Radar_Radius)
count={}
for line in content:
    a=[]
    for t in line.split(','):
	for x in t.split():
            try:
                a.append(float(x))
            except ValueError:
                pass
    scores[(a[2],a[5],a[6],a[7],a[3])]=a[4]
    avg_scores[(a[2],a[5],a[6],a[7])]=0
    count[(a[2],a[5],a[6],a[7])]=0
for params in scores.keys():
   avg_scores[params[:-1]]+=scores[params]
   count[params[:-1]]+=1
for params in avg_scores.keys():
    avg_scores[params]=avg_scores[params]/count[params]

#Part-2 Prepare data from map and CTO actual for the comparisions to be made
x = np.arange(5,30,5)

#Part-3 Serial Genetic Algorithm 

f2=open("output-serial2.txt","r")
content2 = f2.readlines()
scores2={}#Indexed by tuple of (Observation_Count,Target_Count,Radar_Radius,Iteration_Number)
avg_scores2={}#Indexed by tuple of (Observation_Count,Target_Count,Radar_Radius)
count2={}
for line in content2:
    a=[]
    for t in line.split(','):
        for x in t.split():
            try:
                a.append(float(x))
            except ValueError:
                pass
    scores2[(a[2],a[5],a[6],a[7],a[3])]=a[4]
    avg_scores2[(a[2],a[5],a[6],a[7])]=0
    count2[(a[2],a[5],a[6],a[7])]=0
for params in scores2.keys():
    avg_scores2[params[:-1]]+=scores2[params]
    count2[params[:-1]]+=1
for params in avg_scores2.keys():
    avg_scores2[params]=avg_scores2[params]/count2[params]

#Part-4
fig=plt.figure()
for a in range(1,3):
    for b in range(1,3):
        for c in [0.85,0.95]:
            y1=[]
            y3=[]
            x=[]
            for d in [5,10,15,20,25]:
                x.append(d)
                y1.append(avg_scores[(d,a,b,c)])
                y3.append(avg_scores2[(d,a,b,c)]-0.75)
            y2=[2.5,7.5,13,16.5,17.5]
            z=2
            if c==0.85:
                z=1
            ax = plt.figure().add_subplot(1,1,1)#4,2,a+(b-1)*2+(z-1)*4)
            plt.xlabel('Sensor Range')
            plt.ylabel('Mean targets observed')
            selection_strategy="2-Tournament selection"
            crossover_strategy="1-Point Crossover"
            if a==2:
                selection_strategy="3-Tournament selection"
            if b==2:
                crossover_strategy="2-Point Crossover"
            plt.text(1,15, selection_strategy+'\n'+crossover_strategy+'\nCrossover_proability = '+str(c))
            ax.set_xlim([0,27])
            ax.set_ylim([0,24])
            line1, = plt.plot(x,y1,'r',label='Parallel Genetic Algorithm')
            y2=[2.5,7.5,13,16.5,17.5]
            line2, = plt.plot(x,y2,'g',label='CTO K-Means')
            line3, = plt.plot(x,y3,'b',label='Genetic Algorithm')
            plt.legend(handles=[line1, line2, line3])
            #plt.show()
            total=a+(b-1)*2+(z-1)*4
            #plt.savefig('Comparision.png')
            plt.savefig('Comparision-'+str(total)+'.png')

           
#Part-5 Graph plotting
'''
ax = plt.figure().add_subplot(1, 1, 1)
plt.xlabel('Sensor Range')
plt.ylabel('Mean targets observed')
ax.set_xlim([0,27])
ax.set_ylim([0,24])
line1, = plt.plot(x,y1,'r',label='Parallel Genetic Algorithm')
y2=[2.5,7.5,13,16.5,17.5]
line2, = plt.plot(x,y2,'g',label='CTO K-Means')

plt.legend(handles=[line1, line2])
#plt.show()
plt.savefig('Comparision.png')
'''
