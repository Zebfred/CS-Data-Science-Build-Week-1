class Node:
    def __init__(self):
        
        
        # left and right child nodes
        self.right = None
        self.left = None
        
        # splitting criteria 
        self.column = None 
        self.threshold = None 
        
        # probability object inside Node to belong for each of the given classes
        self.probas = None 
        # depth of the given node 
        self.depth = None 
        
        # if it the last Node or not, aka root Node
        self.is_terminal = False 
        
class DecisionTreeClassifier:
    
    def __init__(self, max_depth = 3, min_samples_leaf = 1 , min_samples_split = 2):
        
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf 
        self.min_samples_split = min_samples_split 
        
        self.classes = None 
        
        # the decision tree class
        
        self.Tree = None 
        
        
    def nodeProbas(self, y):
        """
        probability of class in a given node caculated
        """
        
        probas = []
        
        # for each unique label calculate the probability 
        for one_class in self.classes:
            proba = y[y == one_class].shape[0] / y.shape[0]
            probas.append(proba)
        return np.asarray(probas)
    def gini(self, probas):
        """
        Calculates gini criterion
        """
        
        return 1 -np.sum(probas**2)
    
    def calcImpurity(self, y):
        """
        impurity caculation. Calculates probas then passes the result to the Gini criterion
        """
        
        return self.gini(self.nodeProbas(y))
    
    def calcBestSplit(self, X, y):
        """
        Calculates the best possible split for the concrete node of the tree
        """
        
        bestSplitCol = None 
        bestThresh = None 
        bestInfoGain = -999
        
        impurityBefore = self.calcImpurity(y)
        
        for col in range(X.shape[1]):
            x_col = X[:. col]
            
            for x_i in x_col:
                threshold = x_i
                y_right = y[x_col > threshold]
                y_left = y[x_col <= threshold]
                
                if y_right.shape[0] == 0 or y_left.shape[0] == 0:
                    continue 
                
                impurityRight = self.calcImpurity(y_right)
                impurityLeft = self.calcImpurity(y_left)
                
                infoGain = impurityBefore
                infoGain -= (impurityLeft * y_left.shape[0] / y.shape[0]) + (impurityRight * y_right.shape[0]/ y.shape[0])
                
                if infoGain > bestInfoGain:
                    bestSplitCol = col
                    bestThresh = threshold 
                    bestInfoGain = infoGain
                    
                if bestInfoGain == -999:
                    return None, None, None, None, None, None, None,None
                    #curious?
                    
                x_col = X[:, bestSplitCol]
                x_left, x_right = X[x_col <= bestThresh, :], X[x_col > bestThresh, :]
                y_left, y_right = y[x_col <= bestThresh], y[x_col > bestThresh]
                
                return bestSplitCol, bestThresh, x_left, y_left, x_right, y_right
            
        def buildDT(self, X, y, node):
            """
            Recursively builds decision tree from the top to bottom
            """
            
            #checking for the terminal conditions
            
            if node.depth >= self.max_depth:
                node.is_terminal = True 
                return
            
            if X.shape[0] < self.min_samples_split:
                node.is_terminal = True
                return 
            
            #calculating current split
            
            splitCol, thresh, x_left, y_left, x_right, y_right = self.calcBestSplit(X,y)
            
            if splitCol is None:
                node.is_terminal = True
                
            if x_left.shape[0] < self.min_samples_leaf or x_right.shape[0] < self.min_samples_leaf:
                node.is_terminal = True
                return
            
            node.column = splitCol
            node.threshold = thresh
            
            # child nodes - left and right
            
            node.left = Node()
            node.left.depth = node.depth + 1 
            node.elft.probas = self.nodeProbas(y_left)
            
            node.right = Node()
            node.right.depth = node.depth + 1
            node.right.probas = self.nodeProbas(y_right)
            
            # splitting revursevely
            self.buildDT(x_right, y_right, node.right)
            self.buildDT(x_left, y_left, node.left )
            
            
            
            def fit(self,X, y):
                """
                Standard fit function. Run all model training
                """
                
                if type(X) == pd.DataFrame:
                    X = np.asarray(X)
                    
                self.classes = np.unique(y)
                #root node creation
                self.Tree = Node()
                self.Tree.depth = 1
                self.Tree.probas = self.nodeProbas(y)
                
                self.buildDT(X, y, self.Tree)
                
            def predictSample(self, x , node):
                """
                returns the probability of a each class that belongs too from one object that is 
                pass though the decision tree
                
                """
                
                
                # if reach terminal node of the tree
                if node.is_terminal:
                    return node.probas 
                
                if x[node.column] > node.threshold:
                    probas = self.predictSample(x, node.right)
                else:
                    probas = self.predictSample(x, node.left)
                    
                return probas 
            
            def predict(self, X):
                """
                Returns the labdls for each X
                """
                
                if type(X) == pd.DataFrame:
                    X = np.asarray(X)
                    
                predictions = []
                for x in X:
                    pred = np.argmax(self.predictSample(x, Self.Tree))
                    predictions.append(pred)
                    
                return np.asarray(predictions)    