f=[]
f0=open("train.wordpiece.en-fr.0", "w")
f1=open("train.wordpiece.en-fr.0", "w")

no=0
with open("train.wordpiece.en-fr.0", "rb") as f:
    for line in f:
        f[no % 2].write(line)
        no+=1

f0.close()
f1.close()


