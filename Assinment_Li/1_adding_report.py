# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 21:35:16 2022

@author: bobbala
"""

def adding_report(report):
    print("Input an integer to add to the total or 'Q' to quit")

    Total = 0
    items = " "

    while True:
        new_item = input("Enter an integer or Q:")
        if(new_item.isdigit()):
            Total = Total + int(new_item)
            if(report == 'A'):
                items = items + "\n" + new_item 
        elif (new_item.upper() == 'Q'):
            if(report == 'A'):
                print("Entered items:", items)
                print("\nTotal:", Total)
                break
            else: 
                print("\nTotal:", Total)
                break
        else:
            print("Invalid input")

input_ = input("Enter character A, T and Q:")
adding_report(input_)
