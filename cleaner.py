

with open('imagenet_classes_out2.txt', 'r') as infile, \
     open('imagenet_classes_out1.txt', 'w') as outfile:
    data = infile.read()
    data = data.replace("<", "")


    outfile.write(data)