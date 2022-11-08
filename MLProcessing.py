from pandas import read_csv
from matplotlib import pyplot
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
#ASD=0, TD=1
filenames = "C:/Users/Sean/Downloads/Delta power gamma peak_Edited.csv"
class pipeline:
    def __init__(self, algorithmList, filename):
        self.filename = filename
        self.algorithmList = algorithmList
    def doesListContain(self, listToCheck, element):
        for ele in listToCheck:
            if ele == element:
                return True
        return False
    def run(self):
        self.names = read_csv(self.filename, header=None).values[0, :]
        print(self.names)
        self.dataset = read_csv(self.filename,header=0, names=self.names)
        print(self.dataset.head(20))
        print(self.dataset.describe)
        self.X = self.dataset.values[:, 0:self.dataset.values.shape[1]-1]
        self.Y = self.dataset.values[:, self.dataset.values.shape[1]-1]
        print(self.Y)
        print(self.X[:, 130:])
        X_train, X_validation, Y_train, Y_validation = train_test_split(self.X, self.Y, test_size=0.2, random_state=7)
        num_folds=10
        self.dataset.hist()
        self.fig = pyplot.figure()
        self.cax = self.fig.add_subplot(111).matshow(self.dataset.corr(), vmin=-1, vmax=1, interpolation='none')
        self.fig.colorbar(self.cax)
        #Feature Importance
        self.model = ExtraTreesClassifier(n_estimators=100)
        self.model.fit(self.X, self.Y)
        print("Feature Importance-----------------------------")
        for i in range(0, self.names.size-1):
            print("Name" +str(self.names[i]) +" rank" + str(self.model.feature_importances_[i]))

        #RFE
        self.model = LogisticRegression(solver='liblinear')
        self.rfe = RFE(self.model)
        self.fit = self.rfe.fit(self.X, self.Y)
        print("RFE Score--------------------------------------")
        for i in range(0, self.names.size-1):
            print("Name" +str(self.names[i]) +" rank" + str(self.fit.ranking_[i]))
        #print("Selected Features %s" %self.fit.support_)
        #print("Feature Ranking %s" %self.fit.ranking_)

        #UFS
        self.ufs = SelectKBest(score_func=f_classif, k=3)
        self.fit = self.ufs.fit(self.X, self.Y)
        print("UFS Score--------------------------------------")
        for i in range(0, self.names.size-1):
            print("Name: " +str(self.names[i]) +" Rank: " + str(self.fit.scores_[i]))
        #Algorithms
        self.models = []
        if self.doesListContain(self.algorithmList, "LR"):
            self.models.append(('LR', LogisticRegression(solver='liblinear')))
        if self.doesListContain(self.algorithmList, "LDA"):
            self.models.append(('LDA', LinearDiscriminantAnalysis()))
        if self.doesListContain(self.algorithmList, "KNN"):
            self.models.append(('KNN', KNeighborsClassifier()))
        if self.doesListContain(self.algorithmList, "CART"):
            self.models.append(('CART', DecisionTreeClassifier()))
        if self.doesListContain(self.algorithmList, "NB"):
            self.models.append(('NB', GaussianNB()))
        if self.doesListContain(self.algorithmList, "SVM"):
            self.models.append(('SVM', SVC(gamma='auto')))
        if self.doesListContain(self.algorithmList, "AB"):
            self.models.append(('AB', AdaBoostClassifier()))
        if self.doesListContain(self.algorithmList, "GBM"):
            self.models.append(('GBM', GradientBoostingClassifier()))
        if self.doesListContain(self.algorithmList, "RF"):
            self.models.append(('RF', RandomForestClassifier(n_estimators=100)))
        if self.doesListContain(self.algorithmList, "ET"):
            self.models.append(('ET', ExtraTreesClassifier(n_estimators=100)))
            
        self.results = []
        self.names = []
        for name, model in self.models:
            self.kfold = KFold(n_splits=10, random_state=7, shuffle=True)
            self.cv_results = cross_val_score(model, self.X, self.Y, cv=self.kfold)
            self.results.append(self.cv_results)
            
            self.names.append(name)
            msg = "%s: %f (%f)" % (name, self.cv_results.mean(), self.cv_results.std())
            print(msg)
        # Compare Algorithms
        fig = pyplot.figure()
        fig.suptitle('Algorithm Comparison')
        ax = fig.add_subplot(111)
        pyplot.boxplot(self.results)
        ax.set_xticklabels(self.names)

        if self.doesListContain(self.algorithmList, "KNN(Highly Tuned)"):
            # Tune scaled KNN
            scaler = StandardScaler().fit(X_train)
            rescaledX = scaler.transform(X_train)
            neighbors = [1,3,5,7,9,11,13, 15, 17]
            param_grid = dict(n_neighbors=neighbors)
            model = KNeighborsClassifier()
            kfold = KFold(n_splits=10, random_state=7, shuffle=True)
            grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=kfold)
            grid_result = grid.fit(rescaledX, Y_train)
            print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']
            for mean, stdev, param in zip(means, stds, params):
                print("%f (%f) with: %r" % (mean, stdev, param))

        if self.doesListContain(self.algorithmList, "SVM(Highly Tuned)"):
            # Tune scaled SVM
            scaler = StandardScaler().fit(X_train)
            rescaledX = scaler.transform(X_train)
            c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]
            kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
            param_grid = dict(C=c_values, kernel=kernel_values)
            model = SVC(gamma='auto')
            kfold = KFold(n_splits=10, random_state=7, shuffle=True)
            grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=kfold)
            grid_result = grid.fit(rescaledX, Y_train)
            print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']
            for mean, stdev, param in zip(means, stds, params):
                print("%f (%f) with: %r" % (mean, stdev, param))

        # Tune scaled AdaBoost
        if self.doesListContain(self.algorithmList, "AB(Highly Tuned)"):
            scaler = StandardScaler().fit(X_train)
            rescaledX = scaler.transform(X_train)
            n_estimators = [20, 30, 40, 50, 60 ,70, 80, 100]
            learning_rate = [ 0.2, 0.6, 1.0, 2.0]
            algorithm = ['SAMME.R','SAMME']
            param_grid = dict(n_estimators=n_estimators, learning_rate=learning_rate, algorithm=algorithm)
            model = AdaBoostClassifier()
            kfold = KFold(n_splits=10, random_state=7, shuffle=True)
            grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=kfold)
            grid_result = grid.fit(rescaledX, Y_train)
            print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']
            for mean, stdev, param in zip(means, stds, params):
                print("%f (%f) with: %r" % (mean, stdev, param))

        # Tune scaled GBM
        if self.doesListContain(self.algorithmList, "GBM(Highly Tuned)"):
            scaler = StandardScaler().fit(X_train)
            rescaledX = scaler.transform(X_train)
            loss = ['log_loss', 'exponential' ]
            learning_rate = [ 0.1, 0.2, 0.6, 1.0, 2.0]
            n_estimators = [20, 50, 70, 90, 100, 130, 150]
            subsample = [0.1, 0.3, 0.5, 0.7, 1.0, 2.0]
            max_depth = [1, 2, 3, 4, 5]
            #min_samples_split = [2, 4, 10]
            param_grid = dict(loss = loss, learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
            model = GradientBoostingClassifier()
            kfold = KFold(n_splits=10, random_state=7, shuffle=True)
            grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=kfold)
            grid_result = grid.fit(rescaledX, Y_train)
            print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']
            for mean, stdev, param in zip(means, stds, params):
                print("%f (%f) with: %r" % (mean, stdev, param))
            
        pyplot.show()
##p = pipeline([], filenames)
##print(p.doesListContain(["hello", "Me"], "helo"))
##        
