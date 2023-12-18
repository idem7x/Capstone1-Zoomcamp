import requests

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'
data = {'url': 'https://www.kroger.com/product/images/large/front/0000000004062'}

result = requests.post(url, json=data).json()
print(result)
