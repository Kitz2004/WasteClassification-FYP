import pickle
import numpy as np
import os
from waste_classification import (generate_waste_dataset,
                                   preprocess_data,
                                   train_models)

os.makedirs('model', exist_ok=True)

# Generate & Train
df = generate_waste_dataset()
X_train, X_test, y_train, y_test, le, scaler, feature_names = preprocess_data(df)
trained_models = train_models(X_train, y_train)

# Save all
with open('model/random_forest_model.pkl', 'wb') as f:
    pickle.dump(trained_models['Random Forest'], f)
    print("✅ Saved: model/random_forest_model.pkl")

with open('model/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
    print("✅ Saved: model/label_encoder.pkl")

with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
    print("✅ Saved: model/scaler.pkl")

print("\n🎉 ALL MODELS SAVED SUCCESSFULLY!")
print("Now run: python app.py")