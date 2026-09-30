import wandb
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Inicia o projeto no W&B
wandb.init(project="mlops-wandb-intro")

# Carrega os dados e divide em treino/teste
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Treina o modelo
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Calcula a acurácia
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

# Loga a acurácia no dashboard do W&B
wandb.log({"accuracy": acc})
wandb.finish()