T = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y', 'z','1','2','3','4','5','6']

SEQ_LEN = 4
BATCH_SIZE = 2
BATCH_CHARS  = int(len(T) / BATCH_SIZE)

x = [[0,0,0,0],[0,0,0,0]]

print ('Sequence: ', '  '.join(str(c) for c in T))
for i in range(0, BATCH_CHARS - SEQ_LEN +1, SEQ_LEN):
    print ('BATCH', i/SEQ_LEN)
    for batch_idx in range(BATCH_SIZE):
        start = batch_idx * BATCH_CHARS + i          
        print ('\tsequence', batch_idx, 'of batch:',)
        for j in range(SEQ_LEN):
            x[batch_idx][j] = T[start+j]
            print (T[start+j],)
    #here we would yield x (the batch) if this were an iterator
        print