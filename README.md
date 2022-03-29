# kt82
Key Topics - Assignment 8.2



## Environment setup

- python 3.x
- pip
- vscode 
- git 

I've collected (most) of the needed packages in requirements.txt. Install the packages with:

    pip install -r requirements.txt

Testje Martin


## Notes
Remark : I used pip to install opencv-python. This installs a precompiled CPU only version. This is sufficient for this project. GPU is not necessary.

Dataset considerations
- Manual curated dataset. Unbalanced
- 15000 Non trafic ; 5000 Trafic

Save_HOG_LBP code
- It skips around 5000 non-traffic (number 10000-15000).. Potential bug : their code assumes that non-trafic images are listed before traffic images...

Paper mentions that 20.000 images where used. And not about a skip of 5000 records.
I think it is done to get a better balance between Non Traffic and Traffic.

- The NonTrafic/ Trafic names are used in the pickle. Converted to Labels later on.

Classifier
- Paper mentions 80/20 split. 
- Split into test/train is not explicitely made.
- Scikit documentaion says that the `Probably=true` property of the classifier internally uses a 5 fold cross validation, hence the 


## Instructions to replicate the accuracy without LBP 

Remove all the pickles.

Change Save_HOG_LBP to skip the LBP part.
- Change `embedding = np.append(hog_embedding.ravel(),lbp_embedding.ravel())` to `embedding = hog_embedding.ravel()`

Run the Save_HOG_LBP
Run the Classifier


## Used Youtube video 

https://www.youtube.com/watch?v=MNn9qKG2UFI


