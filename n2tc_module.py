import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

class N2TC:
    def __init__(self):
        # Neural Network: Feed-forward back propagation (MLP in sklearn)
        # Paper mentions hidden layers. We use a standard config.
        self.model = MLPClassifier(hidden_layer_sizes=(20, 10), max_iter=500, random_state=42)
        
    def calculate_mathematical_classes(self, df):
        """
        Implementation of Equation (1) and (2) from the paper to generate labels.
        TW[i] = [WP(ET)*ET] + [WP(C)*C] + [WP(SE)*SE]
        
        """
        # Weights for parameters (Hypothetical balanced weights)
        WP_ET = 0.4
        WP_C = 0.3
        WP_SE = 0.3
        
        tasks_classes = []
        
        for index, row in df.iterrows():
            # Eq 1: Task Weight
            TW = (WP_ET * row['ET']) + (WP_C * row['C']) + (WP_SE * row['SE'])
            
            # Eq 2: Classification based on weight thresholds
            # We divide into 3 classes based on quantiles for dynamic grouping
            tasks_classes.append(TW)
            
        # Determine thresholds for Class 1 (High), 2 (Med), 3 (Low)
        t1 = np.percentile(tasks_classes, 33)
        t2 = np.percentile(tasks_classes, 66)
        
        labels = []
        for tw in tasks_classes:
            if tw <= t1: labels.append(1) # Class 1: Light/Fast/Efficient
            elif tw <= t2: labels.append(2)
            else: labels.append(3) # Class 3: Heavy/Costly
            
        return np.array(labels)

    def train(self, df):
        """Trains the Neural Network to recognize task classes."""
        X = df[['ET', 'C', 'SE']]
        y = self.calculate_mathematical_classes(df)
        
        # 70% Train, 15% Val, 15% Test (Simulated via split)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        print("Training N2TC Neural Network...")
        self.model.fit(X_train, y_train)
        print(f"N2TC Training Score: {self.model.score(X_test, y_test):.4f}")
        
    def classify_tasks(self, tasks_df):
        """Predicts class for new tasks."""
        X = tasks_df[['ET', 'C', 'SE']]
        return self.model.predict(X)