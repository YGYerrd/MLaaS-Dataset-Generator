from adaptive_composability import AdaptiveComposabilityModel
import pandas as pd
import numpy as np

# Fake example to confirm wiring
df = pd.DataFrame({
    "DUM": np.random.randint(0, 2, 10),
    "SUM": np.random.randint(0, 2, 10),
    "HQS": np.random.randint(0, 2, 10),
    "SRS": np.random.randint(0, 2, 10),
    "MUM": np.random.randint(0, 2, 10)
})
y_true = np.random.randint(0, 2, 10)

model = AdaptiveComposabilityModel(mode="rule")
results, composed = model.fit_predict(df, y_true)
print(results)
