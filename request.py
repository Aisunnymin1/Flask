import requests

resp = requests.post("http://localhost:5000/predict",
                     files={"file": open('./test/pandas.jpg','rb')})
print(resp.json())