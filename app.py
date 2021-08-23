import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import json

app = Flask(__name__)

app.config['SECRET_KEY'] = 'a really really really really long secret key'

dataset = pd.read_csv("ALL_LAPTOPS.csv")
dataset_c = pd.read_csv("ALL_LAPTOPS.csv")
dataset_c.drop(columns= ['PRODUCT' , 'IMAGE'] , inplace=True)


dataset['BRAND'] = dataset['BRAND'].str.lower()
dataset["PROCESSOR"] = dataset['PROCESSOR'].str.replace(" ", "")
dataset["PROCESSOR"] = dataset['PROCESSOR'].str.lower()
dataset['GRAPHIC_CARD'] = dataset['GRAPHIC_CARD'].str.lower()
dataset['GRAPHIC_CARD'] = dataset['GRAPHIC_CARD'].str.replace(" ", '')
dataset['OPERATING_SYSTEM'] = dataset['OPERATING_SYSTEM'].str.replace(' ','')
dataset['OPERATING_SYSTEM'] = dataset['OPERATING_SYSTEM'].str.lower()
dataset['PRICE'] = dataset['PRICE'].astype(str)
dataset['DISPLAY'] = dataset["DISPLAY"].astype(str)

dataset_product = pd.DataFrame(dataset['PRODUCT'])


product = []

for i in dataset['PRODUCT']:
  temp = i.split('(')[0]
  product.append(temp[:-1])

dataset_laptop = pd.DataFrame(product)

columns = ['PRODUCT']
dataset_laptop.columns = columns
dataset_laptop['IMAGE'] = dataset['IMAGE']
dataset.drop(columns= ['PRODUCT'],inplace=True )
dataset.drop(columns= ['IMAGE'],inplace=True )
lst = []
for k in range(dataset.shape[0]):
   n = [dataset[i][k] for i in dataset.columns]
   n = ' '.join(n)
   lst.append(n)
dataset_spec = pd.DataFrame(lst)
clmn = ['Specification']
dataset_spec.columns = clmn

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(dataset_spec['Specification'])

from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(X , X)

laptop = dataset_product['PRODUCT']

indices = pd.Series(dataset_product.index , laptop)

def recommender(title):
  index = indices[title]
  similarity_score = list(enumerate(similarity[index]))
  similarity_score = sorted(similarity_score , key=lambda x:x[1] , reverse=True)
  similarity_score = similarity_score[1:17]
  lap_indices = [i[0] for i in similarity_score]
  return dataset_product.iloc[lap_indices].values , lap_indices 

def get_suggestions():
  return list(dataset_product['PRODUCT'])

def lap_im_name(laptops):
  lst = recommender(laptops)
  details = []
  details1 = []
  for values in lst[1]:
    details1.append((dataset_c['BRAND'][values] , dataset_c['PROCESSOR'][values] , dataset_c['RAM'][values] , dataset_c['HDD'][values] , dataset_c['SSD'][values] , dataset_c['GRAPHIC_CARD'][values] , dataset_c['DISPLAY'][values] , dataset_c['OPERATING_SYSTEM'][values] , dataset_c['PRICE'][values] , values))
  for i , pro in enumerate(lst[0]):
    index = indices[pro][0]
    details.append((dataset_laptop['IMAGE'][index] , dataset_laptop['PRODUCT'][index] , details1[i]))
  return details


@app.route("/")
def home():
  return render_template('home.html' , suggestions = get_suggestions())


@app.route('/recommend',methods=['GET' , 'POST'])
def recommend():
  laptops = request.form['laptop']
  index = indices[laptops]
  laptop = dataset_laptop['PRODUCT'][index]
  img = dataset_laptop["IMAGE"][index]
  brand = dataset_c['BRAND'][index]
  processor = dataset_c['PROCESSOR'][index]
  ram = dataset_c['RAM'][index]
  hdd = dataset_c["HDD"][index]
  ssd = dataset_c["SSD"][index]
  graphic = dataset_c['GRAPHIC_CARD'][index]
  display = dataset_c['DISPLAY'][index]
  os = dataset_c['OPERATING_SYSTEM'][index]
  price = dataset_c['PRICE'][index]
  final_pair = lap_im_name(laptops)

  return render_template('index.html',pair = final_pair ,similar = lst , img = img, laptop = laptop , brand = brand, processor = processor , ram = ram , hdd = hdd, ssd = ssd , graphic = graphic , display = display , os = os , price = price)


if __name__ == "__main__":
    app.run(debug=True)
