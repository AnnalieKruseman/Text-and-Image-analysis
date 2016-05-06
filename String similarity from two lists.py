
# coding: utf-8

import json
from watson_developer_cloud import ConceptInsightsV2

concept_insights = ConceptInsightsV2(
    username='6cba26de-ef3b-42ff-8935-5fe083ab32e6',
    password='9qG9Ba5dPDJ6')

accounts = concept_insights.get_accounts_info()
print(json.dumps(accounts, indent=2))

graphs = concept_insights.get_graphs()
print(json.dumps(graphs, indent=2))


# Get keywords from course text

annotations = concept_insights.annotate_text('What is Data Modelling You are aware that data modelling is a skill that not only a database administrator possesses. You think normalization is a term to denote the end of the cold war or you have never made a ERD that was understood by others. In all these cases, this course is suitable for you. The organization for which you work will benefit from the modelling skills and you learn the use of DAMA DMBOK (data modelling is one of its core areas). At the end of this training, participants will have gained knowledge of: Assessing data models (3NF). Conceptual - Logical - Physical models Making a (logical / ERD diagrams) data model (via assignments). The importance of meta data and recording of metadata (data dictionary). BI models (star diagram). The importance of data analysis for good (re) modelling. Modelling within DAMA. The concepts of data modelling. The different models that are made during data modeling. In addition, the participant understands the differences between the conceptual (company), the logical (operational) and the physical (physical) data model.. At the end of this training, participants will be able to: apply the correct model at the right time use different diagram techniques apply the principles of normalization use all possible sources of data communicate with all parties involved')
print(json.dumps(annotations, indent=2))

import json
import sys

#load the data into an element
#data={"test1" : "1", "test2" : "2", "test3" : "3"}

#dumps the json object into an element
json_str_course = json.dumps(annotations)

# print the json string
#print json_str

#load the json to a string
resp_course = json.loads(json_str_course)

#print the resp
#print (resp)

#extract an element in the response
#label = resp['annotations'][0]['concept']['label']

#Create a list with only the labels
#1. create empty list
course_label = []
#2. find all labels and append all labels into the empty list
for each in resp_course['annotations']:
    course_label.append(each['concept']['label'])
    #print each['concept']['label'] 
course_label = '\n'.join(set(course_label))
print course_label

# Save list into csv file

labelfile_course = open('course_label.csv', 'w') # open for 'w'riting
labelfile_course.write(str(course_label)) # write text to file
print ("File is saved", labelfile_course)
labelfile_course.close() # close the file


# Get keywords from course text

annotations = concept_insights.annotate_text('Data Science, Machine Learning, Dimensional Modelling BI, Project Management')
print(json.dumps(annotations, indent=2))

import json
import sys

#load the data into an element
#data={"test1" : "1", "test2" : "2", "test3" : "3"}

#dumps the json object into an element
json_str_cv = json.dumps(annotations)

# print the json string_cv
#print json_str_cv

#load the json to a string
resp_cv = json.loads(json_str_cv)

#print the resp
#print (resp_cv)

#extract an element in the response
#label = resp['annotations'][0]['concept']['label']

#Create a list with only the labels
#1. create empty list
cv_label = []
#2. find all labels and append all labels into the empty list
for each in resp_cv['annotations']:
    cv_label.append(each['concept']['label'])
    #print each['concept']['label'] 
cv_label = '\n'.join(cv_label)
print cv_label

# Save list into csv file

labelfile_cv = open('cv_label.csv', 'w') # open for 'w'riting
labelfile_cv.write(str(cv_label)) # write text to file
print ("File is saved", labelfile_cv)
labelfile_cv.close() # close the file


# # Give matching words from two lists

unq_course_label = set(line.strip() for line in open('course_label.csv'))
unq_cv_label = set(line.strip() for line in open('cv_label.csv'))

for line in unq_course_label & unq_cv_label:
    if line:
        print line


# # Give similarity of two lists

import re, math
from collections import Counter

WORD = re.compile(r'\w+')

def get_cosine(vec1, vec2):
     intersection = set(vec1.keys()) & set(vec2.keys())
     numerator = sum([vec1[x] * vec2[x] for x in intersection])

     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)

     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator

def text_to_vector(text):
     words = WORD.findall(text)
     return Counter(words)

vector1 = text_to_vector(course_label)
vector2 = text_to_vector(cv_label)

cosine = get_cosine(vector1, vector2)

print 'Cosine:', cosine

st_1 = 'dogs chase cats'
st_2 = 'dogs hate cats'
# create set of words from string
st_1_wrds = set(st_1.split())
st_2_wrds = set(st_2.split())
# find out the number of unique words in each set
no_wrds_st_1 = len(st_1_wrds)
no_wrds_st_1 = len(st_1_wrds)

# find out the list of common words between the two sets
cmn_wrds = st_1_wrds.intersection(st_2_wrds)
print cmn_wrds
# find out the number of common words between the two sets
no_cmn_wrds = len(st_1_wrds.intersection(st_2_wrds))
print no_cmn_wrds

# get a list of unique words between the two sets
unq_wrds = course_label.union(cv_label)
print unq_wrds
# find the number of unique words between the two sets
no_unq_wrds = len(st_1_wrds.union(st_2_wrds))
print no_unq_wrds

# calculate jaccard similarity
similarity = no_cmn_wrds / (1.0 * no_unq_wrds)
print similarity

