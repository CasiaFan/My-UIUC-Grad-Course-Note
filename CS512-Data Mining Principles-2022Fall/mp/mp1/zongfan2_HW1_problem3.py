from itertools import combinations

enc = {'Cheese': 1, 'Ghee': 2, 'Tea Powder': 3, 'Bread': 4, 'Butter':5, 'Milk':6, 'Lassi':7, 'Yougurt':8, 'Sweet':9, 'Panner':10, 'Coffee Powder':11, 'Sugar':12}

dec = {1: 'Cheese', 2: 'Ghee', 3: 'Tea Powder', 4: 'Bread', 5:'Butter', 6:'Milk', 7:'Lassi', 8:'Yougurt', 9:'Sweet', 10:'Panner', 11:'Coffee Powder', 12: 'Sugar'}


def load_data(data_file):
    data = {}
    counter = 0
    itemset = set([])
    with open(data_file, "r") as f:
        for line in f:
            # transaction = [enc[i] for i in line.strip().split(",")]
            transaction = line.strip().split(",")
            data[counter] = transaction
            counter += 1
            itemset = set(itemset).union(set(transaction))
    # itemset = list(itemset)
    itemset = [tuple([i]) for i in itemset]
    return itemset, data

def join_patterns(itemset):
    newset = []
    item_length = len(itemset[0])
    for i in range(len(itemset)-1):
        for j in range(i+1, len(itemset)):
            item1 = itemset[i]
            item2 = itemset[j]
            pat = tuple(set(item1).union(set(item2)))
            pat_comb = combinations(pat, item_length)
            for subpat in pat_comb:
                if tuple(subpat) not in itemset:
                    break
            if pat not in newset:
                newset.append(pat)
    return newset
        


def check_pattern_frequent(pattern, data, min_sup):
    fil_pattern = []
    sups = []
    for p in pattern:
        sup = 0
        for i in range(len(data)):
            transaction = data[i]
            match = True
            for item in p:
                if isinstance(item, tuple):
                    item = item[0]
                if item not in transaction:
                    match = False
                    break
            if match:
                sup += 1
        sups.append(sup)
        if sup >= min_sup:
            fil_pattern.append(p)
    # print(len(pattern), len(fil_pattern), sups)
    return fil_pattern
                


def run_apriori(data_file, min_sup=1264):
    itemset, data = load_data(data_file)
    all_pattern = []
    new_pattern = itemset
    while True:
        fil_pattern = check_pattern_frequent(new_pattern, data, min_sup)
        # print(fil_pattern)
        if not len(fil_pattern):
            break
        all_pattern += fil_pattern
        new_pattern = join_patterns(fil_pattern)
    print("All frequent patterns: {} (#={})".format(all_pattern, len(all_pattern)))


    

if __name__ == "__main__":
    data_file = "purchase_history.csv"
    min_sup = 1264
    # itemset, data = load_data(data_file)
    # print(len(itemset), data[1])
    # item = [('Lassi',), ('Sugar',), ('Butter',), ('Bread',), ('Sweet',), ('Milk',), ('Cheese',), ('Coffee Powder',), ('Ghee',), ('Yougurt',), ('Panner',), ('Tea Powder',)]
    # it2 = join_patterns(item)
    # print(len(it2))
    # pat = check_pattern_frequent(item, data, 1264)
    # print(len(pat))
    run_apriori(data_file, min_sup=min_sup)