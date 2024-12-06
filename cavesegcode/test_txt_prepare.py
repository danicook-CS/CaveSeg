'''
Takes the path to image folder and saves their names in a .txt file
'''
import os
import sys

path_testset = "/blue/mdjahiduislam/adnanabdullah/mmsegmentation/data/Test/Test_Images/"

file_to_write = "/blue/mdjahiduislam/adnanabdullah/mmsegmentation/demo/test_caveseg.txt"

with open(file_to_write, "w") as a:
    for path, subdirs, files in os.walk(path_testset):
       for filename in sorted(files):
         input_dir = os.path.join(path_testset, filename)
         output_dir = "/blue/mdjahiduislam/adnanabdullah/mmsegmentation/data/Test/outputs/"+"output_caveseg/"+filename
         a.write(str(input_dir)+","+str(output_dir)+os.linesep)

print('text file saved')
