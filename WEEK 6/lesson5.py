import numpy as np
from sklearn.ensemble import RandomForestClassifier

X = np.random.rand(100, 256)
y = np.random.randint(0, 2, 100)

model = RandomForestClassifier()
model.fit(X, y)

new_image = np.random.rand(1, 256)
prediction = model.predict(new_image)
print("Tumor detected!" if prediction[0] == 1 else "No tumor detected.")