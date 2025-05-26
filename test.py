import re

string = '1500.67 sqft +/-'

out = re.sub('[^0-9.]+', '', string)
print(out)