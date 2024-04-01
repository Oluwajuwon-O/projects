from fastapi import FastAPI
import uvicorn

app = FastAPI()

# defining path operation for root endpoint
@app.get('/')
def main():
    return {'message': 'Welcome to my first api'}

# defining path operation for /name endpoint
@app.get('/{name}')
def hello_name(name: str):
    # defining a function that takes only string as input and 
    # outputs the following message
    return {'message':f'Welcome to my first api, {name}'}


from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB

iris = load_iris()

X = iris.data
Y = iris.target

clf = GaussianNB()
clf.fit(X,Y)

from pydantic import BaseModel

class request_body(BaseModel):
    sepal_length : float
    sepal_width : float
    petal_length : float
    petal_width : float
    

@app.post('/predict')
def predict(data : request_body):
    test_data = [[data.sepal_length,
                  data.sepal_width,
                  data.petal_length,
                  data.petal_width
                  ]]
    class_idx = clf.predict(test_data)[0]
    return{'class': iris.target_names[class_idx]}