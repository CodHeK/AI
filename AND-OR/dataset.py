import numpy as np
import random
f = open('test.txt', 'w')

data = []
test_set = []

for i in range(3000):
    if i%2 == 0:
        data.append(np.random.uniform(-0.05, 0.05))
    else:
        data.append(np.random.uniform(0.95, 1.05))

random.shuffle(data)

for i in range(0, 3000, 3):
    test_set.append([ data[i], data[(i+1)], data[i+2] ])
    if data[i]>0.5 or data[i+1]>0.5 or data[i+2]>0.5:
        toappend=1.0
    else:
        toappend=0.0
    f.write(str(data[i])+' '+str(data[(i+1)])+' '+str(data[i+2])+' '+str(toappend)+'\n')
f.close()

# for i in range(0, 3000, 2):
#     test_set.append([ data[i], data[(i+1)] ])
#     if data[i]>0.5 or data[i+1]>0.5:
#         toappend=1.0
#     else:
#         toappend=0.0
#     f.write(str(data[i])+' '+str(data[(i+1)])+' '+str(toappend)+'\n')
# f.close()
