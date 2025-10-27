from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("iris-classification")
os.makedirs("models",exist_ok=True)

def main():

    df=load_iris()
    x=df.data
    y=df.target

    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=43,test_size=0.2)
    n_estimators=50
    model=RandomForestClassifier(n_estimators=n_estimators,random_state=42)
    model.fit(x_train,y_train)

    pred=model.predict(x_test)
    accuracy=accuracy_score(y_test,pred)

    mlflow.log_param('n_estimators',n_estimators)
    mlflow.log_metric('accuracy',accuracy)
    mlflow.sklearn.log_model(model,"model")

    print(f"accuracy{accuracy}")

    joblib.dump(model,"models/model.pkl")
    mlflow.log_artifact("models/model.pkl")

if __name__ == "__main__":
    main()
