import os

NAME = "clement"
PATH = "Dataset/"+NAME.capitalize()+"/"

ls_dir = os.listdir(PATH)

for i, name in enumerate(ls_dir):
    print(PATH+name," -> ",PATH+NAME.lower(),"-{}.jpg".format(i))
    os.rename(PATH+name,PATH+NAME.lower()+"-{}.jpg".format(i))