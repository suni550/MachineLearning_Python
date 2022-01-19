# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 17:48:27 2022

@author: bobbala
"""


def get_names():
    element_names = []
    count = 0
    print("list any 5 of the first 20 elements in the Period table")
    while (count < 5):
        element_name = input("enter an element:")
        if element_name == "":
            print("invalid entry")
        elif element_name in element_names:
            print("duplicate entry")
        else:
            element_names.append(element_name)
            count += 1
            
        # if element_name not in element_names and element_name != "":
        #     element_names.append(element_name)
        #     count += 1
        # elif element_name == "":
        #     print("invalid entry")
        # else: 
        #     print("duplicate item")

    return element_names

#get_ipython().system(u'curl https://raw.githubusercontent.com/MicrosoftLearning/intropython/master/elements1_20.txt -as elements1_20.txt')

get_ipython().system(u'curl https://raw.githubusercontent.com/MicrosoftLearning/intropython/master/elements1_20.txt -o elements1_20.txt')

file = open('elements1_20.txt', 'r')
element = file.readline().strip().lower()
elements_list = []

while element:
    elements_list.append(element)
    element = file.readline().strip().lower()

#print(elements_list)

quiz_list = get_names()
correct_elem_list = []
incorrect_elem_list = []

for ele in range (len(quiz_list)):
    if quiz_list[ele].lower() in elements_list: # Converting given element to lower
        correct_elem_list.append(quiz_list[ele])
    else: 
        print("incorrect item condition")
        incorrect_elem_list.append(quiz_list[ele])

print("correct_elem_list:", correct_elem_list)
print("incorrect_elem_list: ",incorrect_elem_list)

print("correct_elem_list:", len(correct_elem_list))
#percentage = (len(correct_elem_list)*100/5)
percentage = (len(correct_elem_list)*20)
print("Correct element percentage:", percentage)
