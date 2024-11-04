import requests

def get_issue(project, number):
  issue = requests.get(f'https://api.github.com/repos/{project}/issues/{number}').json()
  return issue['title'], issue['body']
