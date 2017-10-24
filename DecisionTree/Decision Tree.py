
#do plotting
import pandas as pd
import numpy as np
import math
from collections import Counter
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

lst = []
lst1 = []

class DTNode(object):
    def __init__(self, label, parent_value=None, properties={}, leaf=False):
        self.label = label
        self.children = []
        self.parent_value = parent_value
        self.properties = properties
        self.leaf = leaf

    def resultfromDtree(self, attrs_dict, subset, DTree):
        if self.leaf:
            return self.label
        val = attrs_dict[self.label]
        for node in self.children:
            if val == node.parent_value:
                return node.resultfromDtree(attrs_dict, subset, DTree)

        if subset is None:
            raise ValueError("Invalid property found: {0}".format(val))
        else:
            counts = DTree.attr_counts(subset, DTree.dependent)
            most_common = max(counts, key=lambda k: counts[k])
            return most_common

    def add_child(self, node):
        self.children.append(node)

    def num_children(self):
        return len(self.children)

    def _num_leaves(self):
        if self.leaf:
            return 1
        else:
            return sum(c.count_leaves for c in self.children)

    def _depth(self, init):
        if self.leaf:
            return init
        else:
            return max(c._depth(init + 1) for c in self.children)

class DTree(object):

    def __init__(self, train_data):
        self.root = None
        self.all_attributes = train_data.columns.tolist()

        data = []
        for x in range(0, len(train_data.index) - 1):
            row = dict(zip(self.all_attributes, train_data.iloc[x].values))
            data.append(row)
        self.data = data
        #print(self.data)

        self.dependent = self.all_attributes[-1]
        self.get_distinct_values()
        self.attributes = [a for a in self.all_attributes if a != self.dependent]

    def get_distinct_values(self):
        values = {}
        for attr in self.all_attributes:
            values[attr] = set(r[attr] for r in self.data)
        self.values = values

    def attr_counts(self, subset, attr):
        counts = Counter()
        for row in subset:
            counts[row[attr]] += 1
        return counts

    def filter_subset(self, subset, attr, value):
        return [r for r in subset if r[attr] == value]

    def resultfromDtree(self, attributes, subset):
        if len(attributes) != len(self.attribute_order):
            print (self.attribute_order)
            raise ValueError("supplied attributes do not match data")
        attrs_dict = dict(zip(self.attribute_order, attributes))
        return self.root.resultfromDtree(attrs_dict, subset, self)

    def test_accuracy(self, testdf, subset=None, nodes = None):
        test_data = []
        for x in range(0, len(testdf.index) - 1):
            row = dict(zip(self.all_attributes, testdf.iloc[x].values))
            test_data.append(row)

        correct = 0.  # Keep track of statistics
        for row in test_data:
            formatted = [row[a] for a in self.attributes]
            decision = self.resultfromDtree(formatted, subset)
            if row[self.dependent] == decision:
                correct += 1

        x = correct/len(test_data)

        if nodes is not None:
            #print("% correct: {0} and nodes {1}".format(correct / len(test_data), nodes))
            lst.append(x)
            lst1.append(nodes)
        else:
            print("% correct: {0}".format(correct / len(test_data)))

    def countForValues(self, subset, attr, value, base=False):
        counts = Counter()
        for row in subset:
            if row[attr] == value or base:
                counts[row[self.dependent]] += 1
        return counts

    def set_attributes(self, attributes):
        self.attribute_order = attributes

    def attr_counts(self, subset, attr):
        counts = Counter()
        for row in subset:
            counts[row[attr]] += 1
        return counts

    def count_leaves(self):
        if self.root.leaf:
            return 1
        else:
            return sum(c._num_leaves for c in self.root.children)

    def distinct_values(self):
        values_list = []
        for s in self.values.values():
            for val in s:
                values_list.append(val)
        return values_list

    def check_remaining(self, new_remaining):
        list = new_remaining

        if(len(list) < 0):
            return False;

        if len(lst) > 1 and lst[-1] < lst [-2]:
            lst[-1] = lst[-2]


class ID3(DTree):

    def build_tree(self, testdf, nodes=0, parent_subset=None, parent=None, parent_value=None, remaining=None):
        if parent_subset is None:
            subset = self.data
        else:
            subset = self.filter_subset(parent_subset, parent.label, parent_value)

        if remaining is None:
            remaining = self.attributes

        use_parent = False
        counts = self.attr_counts(subset, self.dependent)

        if not counts:
            subset = parent_subset
            counts = self.attr_counts(subset, self.dependent)
            use_parent = True

        if len(counts) == 1:  # Only one value of self.dependent detected
            node = DTNode(
                label=list(counts.keys())[0],
                leaf=True,
                parent_value=parent_value
            )
        elif not remaining or use_parent:
            # If there are no remaining attributes, label with the most
            # common attribute in the subset.
            most_common = max(counts, key=lambda k: counts[k])
            node = DTNode(
                label=most_common,
                leaf=True,
                parent_value=parent_value,
                properties={'estimated': True}
            )
        else:
            # Calculate max information gain
            igains = []
            for attr in remaining:
                useEntropy = False
                igains.append((attr, self.calc_informGain(subset, attr, useEntropy)))

            max_attr = max(igains, key=lambda a: a[1])
            prePruning = True

            #prePruning for training samples less than 5%
            if prePruning == False  or (prePruning == True and (len(subset)/len(self.data) > 0.05)):
                # Create the decision tree node
                node = DTNode(
                    max_attr[0],
                    properties={'information_gain': max_attr[1]},
                    parent_value=parent_value
                )
            else:
                most_common = max(counts, key=lambda k: counts[k])
                node = DTNode(
                    label=most_common,
                    leaf=True,
                    parent_value=parent_value,
                    properties={'estimated': True}
                )

        if parent is None:
            # Set known order of attributes for dtree decisions
            self.set_attributes(self.attributes)
            self.root = node
        else:
            parent.add_child(node)

        self.test_accuracy(testdf, subset, nodes + 1)

        root = 1 if nodes == 0 else 0

        # if any are left
        new_remaining = remaining[:]
        self.check_remaining(new_remaining)

        if not node.leaf:  # Continue recursing
            #new_remaining = remaining[:]
            # Remove the just used attribute from the remaining list
            new_remaining.remove(node.label)
            for value in self.values[node.label]:
                nodes = self.build_tree(
                            testdf,
                            nodes+1,
                            parent_subset=subset,
                            parent=node,
                            parent_value=value,
                            remaining=new_remaining
                        )

        # plot the graph
        if(root == 1):
            self.plot_graph()

        return nodes

    def calc_informGain(self, subset, attr, useEntropy):
        if(useEntropy):
            gain = self.get_entropy_value(subset)
            counts = self.attr_counts(subset, attr)
            total = float(sum(counts.values()))  # Coerce to float for division
            for value in self.values[attr]:
                gain += -((counts[value] / total) * self.entropy(subset, attr, value))
            return gain
        else:
            gain = self.get_gini_value(subset)
            counts = self.attr_counts(subset, attr)
            total = float(sum(counts.values()))
            for value in self.values[attr]:
                gain += -((counts[value] / total) * self.entropy(subset, attr, value))
            return gain

    def plot_graph(self):
        plt.plot(lst1, lst)
        plt.ylabel("Accuracy")
        plt.xlabel("Nodes")
        plt.show()

    def get_entropy_value(self, subset):
        return self.entropy(subset, self.dependent, None, base=True)

    def get_gini_value(self, subset):
         return self.gini_index(subset, self.dependent, None, base=True)

    def entropy(self, subset, attr, value, base=False):
        counts = self.countForValues(subset, attr, value, base)
        total = float(sum(counts.values()))

        entropy = 0
        for dv in counts:
            proportion = counts[dv] / total
            entropy += -(proportion * math.log(proportion, 2))
        return entropy

    def gini_index(self, subset, attr, value, base=False):
        counts = self.countForValues(subset, attr, value, base)
        total = float(sum(counts.values()))

        gini = 1.0

        for dv in counts:
            proportion = counts[dv]/total
            gini += -proportion**2

        return gini


def stratify(df, train_proportion):

    df_col = df.iloc[:,9]
    df_col=np.array(df_col)

    values = np.unique(df_col)
    train_inds = pd.DataFrame()
    test_inds = pd.DataFrame()

    for value in values:
        value_inds = df[df.iloc[:,9] == value]
        value_inds = shuffle(value_inds)
        n = int(train_proportion*len(value_inds))
        train_inds = train_inds.append(value_inds[:n])
        test_inds = test_inds.append(value_inds[n:])

    return train_inds,test_inds

def main():

    colNames = ['Clump_thick' , 'Uniformity_sz', 'Uniformity_shp', 'Marg_Adhesion', 'Epithelial_sz', 'Bare_Nuclei', 'Chromatin', 'Nucleoli', 'Mitoses', 'Class']
    df = pd.read_csv('hw2_question1.csv', names = colNames)
    count_malignant = len((df[df.loc[:,'Class'] == 4]).index)
    count_benign  = len((df[df.loc[:,'Class'] == 2]).index)

    print(count_malignant)
    print(count_benign)

    traindf, testdf = stratify(df, 0.67)

    id3 = ID3(traindf)
    id3.build_tree(traindf)

    print("\n final accuracy")
    id3.test_accuracy(traindf)

main()