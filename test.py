import requests
import sys

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'
data = {'url': sys.argv[1]}

result = requests.post(url, json=data).json()
print(result)
