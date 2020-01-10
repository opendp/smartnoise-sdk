import random

"""
Code to create an example dataset for testing. c is y-value; 
c is 0 if a is low and is high, and is 1 otherwise
"""

list_a = [random.randint(0, 50) for i in range(1000)]
list_b = [random.randint(0, 50) for i in range(1000)]
with open("example.csv", "w", newline='') as f:
    f.write("a,b,c\n")
    for i in range(len(list_a)):
        a = list_a[i]
        b = list_b[i]
        if a < 25 < b:
            c = 0
        else:
            c = 1
        f.write(str(a)+","+str(b)+","+str(c)+"\n")

