import numpy as np


test = np.random.rand(3,10)
print(test)
test = test[:, test[1,:].argsort()]

print(test)
