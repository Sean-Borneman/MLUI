# MLUI
The Goal of this project is to provide a simple UI for one to interact with the sklearn Machine learning library. 

### Installation
Either click the drop down code menu and "Download ZIP" or use 
```git clone https://github.com/SB510/MLUI```

In order to run the project run the UI.py file if an error occurs you likley don't have the required libraries installed if this is the case run:
```
pip install pandas
pip install matplotlib
pip install sklearn
```
And you should have all the required Libraries

### Data Preperation
The UI expects two things: A header on your dataset, For the varible that you are classifing to be in the **last** column of the dataset i.e.
```
Col1  Col2  Col3  Classifier
 12    12     5        0
 3     10     3        1
```

### Running Machine Learning
1. Select your data ex. "myDataset.csv" (NOTE: you may need to select "all files" in order to see your file
![image](https://github.com/SB510/MLUI/assets/78392801/b164d7a1-8636-4ef4-940b-3d418f5a5dc9)
2. Select which algorithm you want to be performed.
![image](https://github.com/SB510/MLUI/assets/78392801/d8ae5402-4a44-4ba3-b25f-3a88f3ace734)
3. Be sure to Set Evaluation Metric or the Machine Learning algorithsm will not run.
4. RUN!
### Future Additions
  In the future I'd love to add another quick UI for editing Data inputed into the UI and support for Drag & drop style Deep Learning Network construction
  
