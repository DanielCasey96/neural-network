import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def load_data():
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist['data'], mnist['target']
    y = y.astype(np.uint8)
    X = X / 255.0
    X = np.array(X)
    y = np.array(y)
    return X, y


def relu(Z):
    return np.maximum(0, Z)


def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return expZ / np.sum(expZ, axis=1, keepdims=True)


def initialize_parameters(input_size, hidden_size, output_size):
    W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
    b2 = np.zeros((1, output_size))
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}


def forward_propagation(X, parameters, cache=None):
    if cache is None:
        Z1 = np.dot(X, parameters['W1']) + parameters['b1']
        A1 = relu(Z1)
        Z2 = np.dot(A1, parameters['W2']) + parameters['b2']
        A2 = softmax(Z2)
        cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}
    else:
        A2 = cache['A2']
    return A2, cache


def backward_propagation(X, y, parameters, cache):
    m = X.shape[0]
    dZ2 = cache['A2'] - (np.arange(cache['A2'].shape[1]) == y[:, None])
    dW2 = np.dot(cache['A1'].T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    dA1 = np.dot(dZ2, parameters['W2'].T)
    dZ1 = dA1 * (cache['Z1'] > 0)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    grads = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
    return grads


def update_parameters(parameters, grads, learning_rate=0.01):
    parameters['W1'] -= learning_rate * grads['dW1']
    parameters['b1'] -= learning_rate * grads['db1']
    parameters['W2'] -= learning_rate * grads['dW2']
    parameters['b2'] -= learning_rate * grads['db2']
    return parameters


def compute_loss(Y, probs):
    m = Y.shape[0]
    log_probs = -np.log(probs[range(m), Y])
    loss = np.sum(log_probs) / m
    return loss


def evaluate(X, y, parameters):
    cache = forward_propagation(X, parameters)[1]
    predictions = np.argmax(cache['A2'], axis=1)
    accuracy = np.mean(predictions == y)
    return accuracy


def train_neural_network(X_train, y_train, X_test, y_test, hidden_size, learning_rate, num_epochs, batch_size):
    input_size = X_train.shape[1]
    output_size = 10
    parameters = initialize_parameters(input_size, hidden_size, output_size)

    for epoch in range(num_epochs):
        permutation = np.random.permutation(X_train.shape[0])
        X_train_shuffled = X_train[permutation]
        y_train_shuffled = y_train[permutation]

        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train_shuffled[i:i + batch_size]
            y_batch = y_train_shuffled[i:i + batch_size]

            A2, cache = forward_propagation(X_batch, parameters)
            loss = compute_loss(y_batch, A2)
            grads = backward_propagation(X_batch, y_batch, parameters, cache)
            parameters = update_parameters(parameters, grads, learning_rate)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")
            train_accuracy = evaluate(X_train, y_train, parameters)
            test_accuracy = evaluate(X_test, y_test, parameters)
            print(f"Train Accuracy at Epoch {epoch}: {train_accuracy}")
            print(f"Test Accuracy at Epoch {epoch}: {test_accuracy}")

    return parameters


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    hidden_size = 128
    learning_rate = 0.01
    num_epochs = 50
    batch_size = 32
    parameters = train_neural_network(X_train, y_train, X_test, y_test, hidden_size, learning_rate, num_epochs,
                                      batch_size)
    test_accuracy = evaluate(X_test, y_test, parameters)
    print(f"Final Test Accuracy: {test_accuracy}")


if __name__ == "__main__":
    main()
