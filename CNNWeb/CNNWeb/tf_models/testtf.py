import re

s = 'abcdenfadjsdfhfskd\nkjhsafdjkhksadf\nasadjkhkasdf'
t = re.findall('^a',s,re.M)
print(t)