cur_config = []
counter = 0
for i in range(3,8,1):
    for j in range(2,6,1):
        for k in range(i):
            cur_config.append(2**(j+k))
        print(cur_config)
        cur_config = []
        counter = 1 + counter

print(counter)