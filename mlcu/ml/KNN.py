import cupy as cp

class KNN:
    '''
    KNN模型，支持分类与回归，
    '''
    def __init__(self, task_type='classification'):
        self.train_data = None
        self.train_label = None
        self.label_set = None
        self.task_type = task_type

    def fit(self, train_data, train_label):
        self.train_data = cp.array(train_data)
        if self.task_type == 'regression':
            self.train_label = cp.array(train_label)
        elif self.task_type == 'classification':
            label_set = list(set(train_label))
            self.label_set = label_set
            train_label = cp.array([label_set.index(l) for l in train_label])
            self.train_label = train_label
        else:
            raise ValueError('no correct task')

    def predict(self, test_data, k=3, distance='l2'):
        test_data = cp.array(test_data)
        preds = cp.array([])
        for x in test_data:
            if distance == 'l1':
                dists = self.l1_distance(x)
            elif distance == 'l2':
                dists = self.l2_distance(x)
            else:
                raise ValueError('wrong distance type')
            sorted_idx = cp.argsort(dists)
            knearnest_labels = self.train_label[sorted_idx[:k]]
            pred = None
            if self.task_type == 'regression':
                pred = cp.mean(knearnest_labels)
            elif self.task_type == 'classification':
                knearnest_labels.sort()
                pred = knearnest_labels[len(knearnest_labels) // 2]
            else:
                raise ValueError('no correct task')
            preds = cp.append(preds, pred)
        if self.task_type == 'classification':
            preds = [self.label_set[int(a)] for a in preds]
        return preds

    def l1_distance(self, x):
        return cp.sum(cp.abs(self.train_data-x), axis=1)

    def l2_distance(self, x):
        return cp.sum(cp.square(self.train_data-x), axis=1)
