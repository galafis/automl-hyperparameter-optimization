FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-c", "from src.models.optimizer import AutoMLOptimizer, ParameterSpace; space = ParameterSpace().add_float('x', -5, 5).add_float('y', -5, 5); automl = AutoMLOptimizer(n_trials=50, method='bayesian'); best = automl.optimize(space, lambda p: -(p['x']**2 + p['y']**2)); print(f'Best score: {automl.best_score:.4f}'); print(f'Best params: {automl.best_params}')"]
