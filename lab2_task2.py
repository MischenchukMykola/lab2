from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from lab2_task1 import X_train
from lab2_task1 import y_train
from lab2_task1 import X_test
from lab2_task1 import y_test
from lab2_task1 import y_test_pred

# Створення SVM з поліноміальним ядром
poly_svm = SVC(kernel='poly', degree=8)
poly_svm.fit(X_train, y_train)
y_pred_poly = poly_svm.predict(X_test)

# Створення SVM з гаусовим ядром
rbf_svm = SVC(kernel='rbf')
rbf_svm.fit(X_train, y_train)
y_pred_rbf = rbf_svm.predict(X_test)

# Створення SVM з сигмоїдальним ядром
sigmoid_svm = SVC(kernel='sigmoid')
sigmoid_svm.fit(X_train, y_train)
y_pred_sigmoid = sigmoid_svm.predict(X_test)

print("Shape of y_test:", y_test.shape)
print("Shape of y_test_pred:", y_test_pred.shape)

# Перевірка помилок під час побудови моделей SVM
if not hasattr(poly_svm, "fit"):
    raise ValueError("SVM model with polynomial kernel was not properly trained.")
if not hasattr(rbf_svm, "fit"):
    raise ValueError("SVM model with RBF kernel was not properly trained.")
if not hasattr(sigmoid_svm, "fit"):
    raise ValueError("SVM model with sigmoid kernel was not properly trained.")

# Обчислення показників якості класифікації для поліноміального SVM
accuracy_poly = accuracy_score(y_test, y_pred_poly)
precision_poly = precision_score(y_test, y_pred_poly)
recall_poly = recall_score(y_test, y_pred_poly)
f1_poly = f1_score(y_test, y_pred_poly)

# Обчислення показників якості класифікації для гаусового SVM
accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
precision_rbf = precision_score(y_test, y_pred_rbf)
recall_rbf = recall_score(y_test, y_pred_rbf)
f1_rbf = f1_score(y_test, y_pred_rbf)

# Обчислення показників якості класифікації для сигмоїдального SVM
accuracy_sigmoid = accuracy_score(y_test, y_pred_sigmoid)
precision_sigmoid = precision_score(y_test, y_pred_sigmoid)
recall_sigmoid = recall_score(y_test, y_pred_sigmoid)
f1_sigmoid = f1_score(y_test, y_pred_sigmoid)


# Виведення результатів

print("\nPolynomial SVM:")
print(f"Accuracy: {accuracy_poly * 100:.2f}%")
print(f"Recall: {recall_poly * 100:.2f}%")
print(f"Precision: {precision_poly * 100:.2f}%")
print(f"F1 Score: {f1_poly * 100:.2f}%")

print("\nGaussian SVM:")
print(f"Accuracy: {accuracy_rbf * 100:.2f}%")
print(f"Recall: {recall_rbf * 100:.2f}%")
print(f"Precision: {precision_rbf * 100:.2f}%")
print(f"F1 Score: {f1_rbf * 100:.2f}%")

print("\nSigmoid SVM:")
print(f"Accuracy: {accuracy_sigmoid * 100:.2f}%")
print(f"Recall: {recall_sigmoid * 100:.2f}%")
print(f"Precision: {precision_sigmoid * 100:.2f}%")
print(f"F1 Score: {f1_sigmoid * 100:.2f}%")
