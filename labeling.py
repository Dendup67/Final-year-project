import os

path = os.chdir("C:/Users/Nova DC/Desktop/Wild boar")

i = 1
for file in os.listdir(path):
    new_file_name = "awake.{}.jpg".format(i)
    os.rename(file, new_file_name)
    i = i+1