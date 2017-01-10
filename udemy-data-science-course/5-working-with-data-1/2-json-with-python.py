import numpy as np
import pandas as pd
import json

from pandas import Series, DataFrame

json_obj = """
{
    "zoo_animal" : "Lion",
    "food" : ["Meat", "Veggies", "Honey"],
    "fur" : "Golden",
    "clothes" : null,
    "diet" : [{"zoo_animal" : "Gazelle", "food" : "grass", "fur" : "Bown"}]
}
"""

data = json.loads(json_obj)
print data

print json.dumps(data)

df = DataFrame(data['diet'])
print df