import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# Đọc dữ liệu
df = pd.read_csv("pose_data.csv")
X = df.drop("label", axis=1).values
y = df["label"].values

# Chia tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Huấn luyện mô hình
clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500)
clf.fit(X_train, y_train)

# Đánh giá
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
