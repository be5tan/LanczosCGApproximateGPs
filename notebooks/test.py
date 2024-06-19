import time

# Start time measurement
start = time.time()
for i in range(5000):
    for j in range(5000):
        sum = i + j 

end = time.time()
print(end - start)
