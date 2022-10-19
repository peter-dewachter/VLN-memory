import csv
import numpy as np

def read(dir):
    with open('tasks/R2R/plots/' + dir + 'seq2seq_sample_log.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        s = []
        sum = 0.0
        vs = []
        vu = []
        for i, row in enumerate(reader):
            if i == 0:
                continue
            #print(row[7])
            seen = float(row[7])
            unseen = float(row[13])
            s.append(seen + unseen)
            if i > 50:
                sum += seen + unseen
            vs.append(seen)
            vu.append(unseen)
        s = np.array(s)
        #print(sum)
        ar = np.argsort(-s, 0)
        for i in range(3):
            print(vs[ar[i]], vu[ar[i]], s[ar[i]] *0.5, ar[i] )
        print(sum)
        print("\n")


if __name__ == "__main__":
    read('dnc/')
    read('C_attn_persist/')



