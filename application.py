from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.preprocessing import normalize
import pickle
import scipy
from scipy.sparse import hstack


# load the model from disk
filename = 'cancerlr.pkl'
clf = pickle.load(open(filename, 'rb'))
gene_vectorizer=pickle.load(open('onehotgene.pkl','rb'))
variation_vectorizer=pickle.load(open('onehotvariation.pkl','rb'))
text_vectorizer=pickle.load(open('onehottext.pkl','rb'))
application = Flask(__name__)

@application.route('/')
def home():
	return render_template('home.html')

@application.route('/predict',methods=['POST','GET'])
def predict():
    

 if request.method == 'POST':
  
  gene = request.form['gene']
  data1 = [gene]
  onehotgene = gene_vectorizer.transform(data1)
  variation=request.form['variation'] 
  data2=[variation]
  onehotvariation=variation_vectorizer.transform(data2)
  text=request.form['text'] 
  data3=[text]
  onehottext=text_vectorizer.transform(data3)
  onehottext=normalize(onehottext,axis=0)
        
  test_gene_var_onehotCoding = hstack((onehotgene,onehotvariation))
  test_onehot=hstack((test_gene_var_onehotCoding,onehottext))
         
  my_prediction = clf.predict(test_onehot)
  pred_dict={1:'Likely Loss-of-function', 2:'Likely Gain-of-function', 3:'Neutral', 4:'Loss-of-function', 5:'Likely Neutral', 6:'Inconclusive', 7:'Gain-of-function', 8:'Likely Switch-of-function', 9:'Switch-of-function'}
 return render_template('home.html',prediction = pred_dict[my_prediction[0]])


if __name__ == '__main__':
	application.run(debug=True)
