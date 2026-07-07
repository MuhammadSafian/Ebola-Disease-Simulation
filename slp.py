import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

class SingleLayerPerceptron:
    """
    Single Layer Perceptron for Ebola infection risk prediction using sklearn.
    """

    def __init__(self, n_inputs=4, learning_rate=0.01):
        self.n_inputs = n_inputs
        self.learning_rate = learning_rate

        # Initialize sklearn's pure Perceptron
        # Using partial_fit later, so max_iter=1 per fit call
        self.model = Perceptron(
            eta0=learning_rate,
            max_iter=1,
            tol=None,
            shuffle=True,
            warm_start=True,
            random_state=42
        )

        # Training history for the UI charts
        self.loss_history = []
        self.accuracy_history = []
        self.epochs_trained = 0
        self.is_trained = False

    def _calculate_perceptron_loss(self, X, y):
        """Manually calculate the perceptron criterion loss for tracking."""
        # Convert true y from {0, 1} to {-1, 1}
        y_signed = np.where(y == 0, -1, 1)
        
        # decision_function gives distance to hyperplane: w^T x + b
        decisions = self.model.decision_function(X)
        
        # Perceptron loss is max(0, -y * f(x))
        loss = np.maximum(0, -y_signed * decisions)
        return float(np.mean(loss))

    def train(self, X, y, epochs=100, verbose=False):
        """
        Train the Perceptron using sklearn's partial_fit to track history.
        """
        self.loss_history = []
        self.accuracy_history = []
        classes = np.array([0, 1])

        for epoch in range(epochs):
            # One pass over the data (no backprop, just perceptron update rule)
            self.model.partial_fit(X, y, classes=classes)

            # Record accuracy
            y_pred = self.model.predict(X)
            acc = accuracy_score(y, y_pred)
            self.accuracy_history.append(float(acc))

            # Record loss
            loss = self._calculate_perceptron_loss(X, y)
            self.loss_history.append(loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs} — Loss: {loss:.4f}, Accuracy: {acc:.4f}")

        self.epochs_trained = epochs
        self.is_trained = True

        if verbose:
            print(f"\nTraining complete! Final accuracy: {self.accuracy_history[-1]:.4f}")

    def _sigmoid(self, z):
        """Map decision function output to 0-1 probability for the UI."""
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def predict_batch(self, X):
        """
        Batch prediction for all agents. 
        Uses sigmoid on the decision_function so we get continuous risk percentages
        rather than just flat 0% or 100%.
        """
        X = np.atleast_2d(X)
        if not self.is_trained:
            return np.full(X.shape[0], 0.5)

        decisions = self.model.decision_function(X)
        return self._sigmoid(decisions)

    def get_info(self):
        """Return SLP information for frontend display."""
        if self.is_trained:
            w = self.model.coef_[0].tolist()
            b = float(self.model.intercept_[0])
        else:
            w = [0.0] * self.n_inputs
            b = 0.0

        return {
            'weights': w,
            'bias': b,
            'accuracy': float(self.accuracy_history[-1]) if self.accuracy_history else 0.0,
            'final_loss': float(self.loss_history[-1]) if self.loss_history else 0.0,
            'loss_history': self.loss_history,
            'accuracy_history': self.accuracy_history,
            'epochs_trained': self.epochs_trained,
            'is_trained': self.is_trained,
            'n_inputs': self.n_inputs,
            'feature_names': ['Age', 'Infected Contacts', 'Healthcare Access', 'Local Crowding']
        }

    def get_weights_str(self):
        """Human-readable weights string for display."""
        if self.is_trained:
            w = self.model.coef_[0]
            b = self.model.intercept_[0]
        else:
            w = [0.0] * self.n_inputs
            b = 0.0
            
        names = ['Age', 'InfContacts', 'Health', 'Crowding']
        parts = [f"{names[i]}: {w[i]:.4f}" for i in range(self.n_inputs)]
        return " | ".join(parts) + f" | Bias: {b:.4f}"

# Quick test
if __name__ == '__main__':
    print("Testing sklearn SingleLayerPerceptron...")
    np.random.seed(42)
    n = 200
    X_test = np.random.rand(n, 4)
    y_test = (X_test[:, 0] > 0.5).astype(float)

    slp = SingleLayerPerceptron(n_inputs=4, learning_rate=0.1)
    slp.train(X_test, y_test, epochs=100, verbose=True)

    info = slp.get_info()
    print(f"\nWeights: {slp.get_weights_str()}")
    print(f"Accuracy: {info['accuracy']:.4f}")

    batch = np.random.rand(5000, 4)
    risks = slp.predict_batch(batch)
    print(f"\nBatch prediction (5000 agents): min={risks.min():.4f}, max={risks.max():.4f}, mean={risks.mean():.4f}")
