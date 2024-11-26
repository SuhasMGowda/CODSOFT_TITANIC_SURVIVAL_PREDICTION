import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

file_path = r'C:\Users\ACER\Desktop\Suhas\Internship\CodeSoft\Titanic-Dataset.csv'
data = pd.read_csv(file_path)

data = data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

sex_mapping = {'male': 0, 'female': 1}
embarked_mapping = {'C': 0, 'Q': 1, 'S': 2}
data['Sex'] = data['Sex'].map(sex_mapping)
data['Embarked'] = data['Embarked'].map(embarked_mapping)

X = data.drop(columns=['Survived'])
y = data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

def predict_survival():
    print("Enter passenger details:")
    try:
        pclass = int(input("Passenger Class (1, 2, or 3): "))
        sex = input("Sex (male or female): ").strip().lower()
        age = float(input("Age: "))
        sibsp = int(input("Number of Siblings/Spouses Aboard: "))
        parch = int(input("Number of Parents/Children Aboard: "))
        fare = float(input("Fare Amount: "))
        embarked = input("Port of Embarkation (C, Q, or S): ").strip().upper()

        sex = sex_mapping[sex]
        embarked = embarked_mapping[embarked]

        input_data = pd.DataFrame({
            'Pclass': [pclass],
            'Sex': [sex],
            'Age': [age],
            'SibSp': [sibsp],
            'Parch': [parch],
            'Fare': [fare],
            'Embarked': [embarked]
        })

        prediction = model.predict(input_data)[0]
        print("\nPrediction: The passenger " + ("survived." if prediction == 1 else "did not survive."))

    except KeyError as e:
        print(f"Invalid input for {e}. Please check the values.")
    except Exception as e:
        print(f"Error: {e}. Please try again.")

predict_survival()