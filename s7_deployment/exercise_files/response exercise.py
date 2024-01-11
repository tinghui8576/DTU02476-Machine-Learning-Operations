import requests
def check_status(response):
    if response.status_code == 200:
        print('Success!')
    elif response.status_code == 404:
        print('Not Found.')
# Check connection status
response = requests.get('https://api.github.com/this-api-should-not-exist')
check_status(response)
response = requests.get('https://api.github.com')
check_status(response)

# See the requests type
response=requests.get("https://api.github.com/repos/SkafteNicki/dtu_mlops")
print(type(response.content))
# Transfer to dictionary
response = requests.get(
    'https://api.github.com/search/repositories',
    params={'q': 'requests+language:python'},
)
print(type(response.json()))

# To download an image
response = requests.get('https://imgs.xkcd.com/comics/making_progress.png')
# cannot be converted into JSON since it's already in byte(remove the comments)
# print(response.json())
# save content
with open(r'img.png','wb') as f:
    f.write(response.content)

# Sending data to the server
pload = {'username':'Olivia','password':'123'}
response = requests.post('https://httpbin.org/post', data = pload)
print(response)
