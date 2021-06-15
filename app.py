from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer


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
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])

def predict():
    

 if request.method == 'POST':
  gene = request.form['gene']
  data1 = [gene]
  onehotgene = gene_vectorizer.transform(data1)
  variation=request.form['variation'] 
  data2=[variation]
  onehotvariation=variation_vectorizer.transformer(data2)
  text=request.form['text'] 
  data3=[text]
  onehottext=text_vectorizer.transformer(data3)
  onehottext=normalize(onehottext,axis=0)
        
  test_gene_var_onehotCoding = hstack((onehotgene,onehotvariation))
  test_onehot=hstack((test_gene_var_onehotCoding,onehottext))
         
  my_prediction = clf.predict(test_onehot)
 return render_template('result.html',prediction = my_prediction)


if __name__ == '__main__':
	app.run(host='0.0.0.0',port=8080)
