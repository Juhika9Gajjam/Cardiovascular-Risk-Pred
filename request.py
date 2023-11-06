import requests
from flask import Flask, request
url = 'http://localhost:5000'
li=[]
for i in request.form.values():
    i=str(i)
    if i.isalpha()==True:
        if i.lower()=='m' or i.lower()=='yes':
            li.append(1)
        else:
            li.append(0)
    else:
        li.append(float(i))
features = [li]
r = requests.post(url,json={'age':features[0], 'education':features[1], 'cigsPerDay':features[2], 'BPMeds':features[3],'prevalentStroke':features[4], 'prevalentHyp':features[5],'diabetes':features[6],'totChol':features[7],'sysBP':features[8], 'diaBP':features[9],'BMI':features[10],'heartRate':features[11], 'glucose':features[12],'sex':features[13],'is_smoking':features[14]})

print(r.json())