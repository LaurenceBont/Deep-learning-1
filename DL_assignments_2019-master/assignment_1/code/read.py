import pickle

with open ('outfile', 'rb') as fp:
    itemlist = pickle.load(fp)

print(itemlist)