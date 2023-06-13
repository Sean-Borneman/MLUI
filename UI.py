from pandas import read_csv
from matplotlib import pyplot
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from MLProcessing import pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from tkinter import *
  
# import filedialog module
from tkinter import filedialog

# Create the root window
window = Tk()
# Set window title
window.title('File Explorer')
# Set window size
window.geometry("920x1080")
#Set window background color
window.config(background = "white")

global filename
filename = "not Set"
# Function for opening the
# file explorer window
def Next(win):
    print("I've been run")
    pass
def RunML(baselineMLbox, filename):
    print("YOU JUST RAN ME")
    algorithms = []
    for i in baselineMLbox.curselection():
        algorithms.append(baselineMLbox.get(i))
    print(clicked.get())
    print(filename)
    print(algorithms)
    p = pipeline(algorithms, filename)
    p.run()
    
def browseFiles():
    print("I was run")
    global filename
    filename = filedialog.askopenfilename(initialdir = "/",
                                          title = "Select a File",
                                          filetypes = (("Text files",
                                                        "*.txt*"),
                                                       ("all files",
                                                        "*.*")))
    # Change label contents
    label_file_explorer.configure(text="File Opened: "+filename)
    #p = pipeline(filename)
    #p.run()

    

# Create a File Explorer label
label_file_explorer = Label(window, text = "Please Select a File", width = 100, height = 4, fg = "blue")
button_explore = Button(window,text = "Browse Files",command = browseFiles)    
button_exit = Button(window, text = "Exit", command = exit)
button_next = Button(window, text = "Next", command = lambda: Next(window))

listbox = Listbox(window, width=40, height=20, selectmode=MULTIPLE)
 
# Inserting the listbox items
listbox.insert(1, "UFS(Feature Importance)")
listbox.insert(2, "RF(Feature Importance)")
listbox.insert(3, "RFE(Feature Importance)")
listbox.insert(4, "Correlation Matrix")
listbox.insert(5, "Histogram")
listbox.insert(6, "LR")
listbox.insert(7, "LDA")
listbox.insert(8, "KNN")
listbox.insert(9, "CART")
listbox.insert(10, "NB")

listbox.insert(11, "SVM")
listbox.insert(12, "AB")
listbox.insert(13, "GBM")
listbox.insert(14, "RF")
listbox.insert(15, "ET")
# Inserting the listbox items
listbox.insert(16, "AB(Highly tuned)")
listbox.insert(17, "SVM(Highly Tuned)")
listbox.insert(18, "KNN(Highly Tuned)")
listbox.insert(19, "GBM(Highly Tuned)")

evalOptions = ["Accuracy", "K-Fold-10(Accuracy and SDev", "ROF", "LogLoss", "AUC", "Classification Report", "Confusion Matrix" ]
clicked = StringVar()
clicked.set("Set Evaluation Metric for Core ML Algorithms")
evalOptionsMenu = OptionMenu(window, clicked, *evalOptions)
guidanceBaseline = Text(window,height=2,width=40)
guidanceBaseline.insert('end', "Which processing algorithsm do you want to use")
guidanceBaseline.config(state='disabled')
guidanceIntro = Text(window, height = 9, width = 80)
guidanceIntro.insert('end', "Here you can select your data file and the algorithms you want to use to classify it. \n 1. This program is meant only for classification not regression \n 2. Format your data file: \n the parameter you want to classify must be in the last column of the datasetall of the parameters in the dataset must be numbers, for strings simply convert them to integers for example if you have state1 and state2 that can be replaced with 0 and 1")
guidanceIntro.config(state='disabled')
guidanceTuned = Text(window, height = 2, width = 40)
guidanceTuned.insert('end', "Which algorithms do you want to be Tuned (finds the best values)");
guidanceTuned.config(state='disabled')
button_train=Button(window, text="RUN!(This will take a while do not turn off your PC, hit cntrl+c if you want to stop the process)", command = lambda: RunML(listbox, filename))




guidanceIntro.grid(column = 0, row = 1)
guidanceTuned.grid(column=0, row =7)
guidanceBaseline.grid(column = 0, row =5)
listbox.grid(column=0, row=6)
label_file_explorer.grid(column = 0, row = 2)
button_explore.grid(column = 0, row = 3)
button_exit.grid(column = 0,row = 4)
button_train.grid(column=0, row = 10)
evalOptionsMenu.grid(column = 0,row = 9)
# Let the window wait for any events
window.mainloop()
