import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.naive_bayes import GaussianNB

def preprocess_data():
    """
    Functie cu scopul de a preprocesa datele necesare:
    - citirea fisierelor excel pentru a salva dataseturile
    - modificarea colonaelor in elemente numerice pentru consistenta
    - modificarea coloanei Data in DateTime
    - extragere elemente temporale din Data pentru gruparea instantelor
    - eliminarea instantelor incomplete (NaN)
    :return: datasetul de antrenare si datasetul de validare
    """
    training_file_path = "datasets/training_data.xlsx"
    test_file_path = "datasets/test_data.xlsx"
    training_data = pd.read_excel(training_file_path, sheet_name="Grafic SEN")
    test_data = pd.read_excel(test_file_path, sheet_name="Grafic SEN")

    columns = [
        'Consum[MW]', 'Medie Consum[MW]', 'Productie[MW]', 'Carbune[MW]',
        'Hidrocarburi[MW]', 'Ape[MW]', 'Nuclear[MW]',
        'Eolian[MW]', 'Foto[MW]', 'Biomasa[MW]', 'Sold[MW]'
    ]

    for col in columns:
        training_data[col] = pd.to_numeric(training_data[col], errors='coerce')
        test_data[col] = pd.to_numeric(test_data[col], errors="coerce")

    training_data['Data'] = pd.to_datetime(training_data['Data'], format='%d-%m-%Y %H:%M:%S', errors='coerce')
    test_data['Data'] = pd.to_datetime(test_data['Data'], format='%d-%m-%Y %H:%M:%S', errors='coerce')

    for df in [training_data, test_data]:
        df['Hour'] = df['Data'].dt.hour
        df['Day'] = df['Data'].dt.day
        df['Month'] = df['Data'].dt.month
        df['Year'] = df['Data'].dt.year
        df['DayOfWeek'] = df['Data'].dt.dayofweek

    training_data = training_data.dropna()
    test_data = test_data.dropna()

    return training_data, test_data

def sold_only_prediction(training_data, test_data, bucket_amount, tree_depth, include_december_only):
    """
    Metoda 1: prezicerea coloanei Sold[MW] fara alte coloane folosind ID3 si Bayes adaptati pentru regresie
    :param training_data: datasetul de antrenare
    :param test_data: datasetul de validare
    :param bucket_amount: numarul de buckets pentru aceasta iteratie
    :param tree_depth: adancimea maxima pentru ID3 in cadrul acestei iteratii
    :param include_december_only: boolean care specifica daca folosim doar lunile decembrie
    :return: graficul cu rezultatul acestei iteratii
    """

    training_data = training_data[['Hour', 'Day', 'Month', 'Year', 'DayOfWeek', 'Data', 'Sold[MW]']].copy()
    test_data = test_data[['Hour', 'Day', 'Month', 'Year', 'DayOfWeek', 'Data', 'Sold[MW]']].copy()

    if include_december_only:
        training_data = training_data[training_data['Data'].dt.month == 12].copy()

    training_data['Bucket'] = pd.cut(training_data['Sold[MW]'], bins=bucket_amount, labels=False)
    training_target = training_data[['Bucket']]
    bucket_edges = pd.cut(training_data['Sold[MW]'], bins=bucket_amount).cat.categories
    bucket_means = bucket_edges.mid.values
    real_values = test_data['Sold[MW]'].values

    # -- ID3 --
    id3_training_features = training_data[['Hour', 'Day', 'Month', 'Year', 'DayOfWeek']]
    id3_test_features = test_data[['Hour', 'Day', 'Month', 'Year', 'DayOfWeek']]

    id3_model = DecisionTreeClassifier(criterion='entropy', max_depth=tree_depth, random_state=42)
    id3_model.fit(id3_training_features, training_target)

    id3_bucket_predictions = id3_model.predict(id3_test_features)
    id3_value_predictions = [bucket_means[bucket] for bucket in id3_bucket_predictions]

    # -- Bayes --
    for col in ['Hour', 'Day', 'Month', 'Year', 'DayOfWeek']:
        training_data[f'{col}Bucket'] = pd.qcut(training_data[col], q=bucket_amount, labels=False, duplicates='drop').fillna(-1)
        test_data[f'{col}Bucket'] = pd.cut(test_data[col], bins=bucket_amount, labels=False).fillna(-1)

    bayes_training_features = training_data[[f'{col}Bucket' for col in ['Hour', 'Day', 'Month', 'Year', 'DayOfWeek']]]
    bayes_test_features = test_data[[f'{col}Bucket' for col in ['Hour', 'Day', 'Month', 'Year', 'DayOfWeek']]]

    bayes_model = GaussianNB()
    bayes_model.fit(bayes_training_features, training_target)

    bayes_bucket_predictions = bayes_model.predict(bayes_test_features)
    bayes_value_predictions = [bucket_means[bucket] for bucket in bayes_bucket_predictions]

    id3_rmse = mean_squared_error(real_values, id3_value_predictions) ** 0.5
    id3_mae = mean_absolute_error(real_values, id3_value_predictions)

    bayes_rmse = mean_squared_error(real_values, bayes_value_predictions) ** 0.5
    bayes_mae = mean_absolute_error(real_values, bayes_value_predictions)

    print(f'Performanta ID3: RMSE = {id3_rmse}, MAE = {id3_mae}')
    print(f'Performanta Bayes: RMSE = {bayes_rmse}, MAE = {bayes_mae}')

    plt.figure(figsize=(12, 7))
    plt.plot(test_data['Data'], real_values, label='Valori reale', color='red')
    plt.plot(test_data['Data'], id3_value_predictions, label='Predictii ID3', color='blue')
    plt.plot(test_data['Data'], bayes_value_predictions, label='Predictii Bayes', color='green')
    plt.xlabel('Data')
    plt.ylabel('Sold[MW]')
    plt.title('Grafic rezultate')
    plt.legend()
    plt.show()

def all_features_prediction(training_data, test_data, bucket_amount, tree_depth, include_december_only):
    """
    Metoda 2: prezicerea coloanei Sold[MW] folosind toate coloanele cu ID3 si Bayes adaptati pentru regresie
    :param training_data: datasetul de antrenare
    :param test_data: datasetul de validare
    :param bucket_amount: numarul de buckets pentru aceasta iteratie
    :param tree_depth: adancimea maxima pentru ID3 in cadrul acestei iteratii
    :param include_december_only: boolean care specifica daca folosim doar lunile decembrie
    :return: graficul cu rezultatul acestei iteratii
    """

    training_data = training_data.copy()
    test_data = test_data.copy()

    if include_december_only:
        training_data = training_data[training_data['Data'].dt.month == 12].copy()

    features = [
        'Consum[MW]', 'Productie[MW]', 'Carbune[MW]',
        'Hidrocarburi[MW]', 'Ape[MW]', 'Nuclear[MW]',
        'Eolian[MW]', 'Foto[MW]', 'Biomasa[MW]'
    ]

    id3_predictions = {}
    bayes_predictions = {}

    real_values = test_data['Sold[MW]'].values

    for feature in features:
        id3_training_target = bayes_training_target = training_data[feature]

        # -- ID3 --
        id3_training_features = training_data[['Hour', 'Day', 'Month', 'Year', 'DayOfWeek']]
        id3_test_features = test_data[['Hour', 'Day', 'Month', 'Year', 'DayOfWeek']]

        id3_model = DecisionTreeClassifier(criterion='entropy', max_depth=tree_depth, random_state=42)
        id3_model.fit(id3_training_features, id3_training_target)

        # -- Bayes --
        for col in ['Hour', 'Day', 'Month', 'Year', 'DayOfWeek']:
            training_data[f'{col}Bucket'] = pd.qcut(training_data[col], q=bucket_amount, labels=False, duplicates='drop').fillna(-1)
            test_data[f'{col}Bucket'] = pd.cut(test_data[col], bins=bucket_amount, labels=False).fillna(-1)

        bayes_training_features = training_data[[f'{col}Bucket' for col in ['Hour', 'Day', 'Month', 'Year', 'DayOfWeek']]]
        bayes_test_features = test_data[[f'{col}Bucket' for col in ['Hour', 'Day', 'Month', 'Year', 'DayOfWeek']]]

        bayes_model = GaussianNB()
        bayes_model.fit(bayes_training_features, bayes_training_target)

        id3_predictions[feature] = id3_model.predict(id3_test_features)
        bayes_predictions[feature] = bayes_model.predict(bayes_test_features)

    id3_value_predictions = abs(id3_predictions['Productie[MW]'] - id3_predictions['Consum[MW]'])
    bayes_value_predictions = abs(bayes_predictions['Productie[MW]'] - bayes_predictions['Consum[MW]'])

    id3_rmse = mean_squared_error(real_values, id3_value_predictions) ** 0.5
    id3_mae = mean_absolute_error(real_values, id3_value_predictions)

    bayes_rmse = mean_squared_error(real_values, bayes_value_predictions) ** 0.5
    bayes_mae = mean_absolute_error(real_values, bayes_value_predictions)

    print(f'Performanta ID3: RMSE = {id3_rmse}, MAE = {id3_mae}')
    print(f'Performanta Bayes: RMSE = {bayes_rmse}, MAE = {bayes_mae}')

    plt.figure(figsize=(12, 8))
    plt.plot(test_data['Data'], real_values, label='Valori reale', color='red')
    plt.plot(test_data['Data'], id3_value_predictions, label='Predictii ID3', color='blue')
    plt.plot(test_data['Data'], bayes_value_predictions, label='Predictii Bayes', color='green')
    plt.xlabel('Data')
    plt.ylabel('Sold[MW]')
    plt.title('Grafic rezultate')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    training_data, test_data = preprocess_data()
    #sold_only_prediction(training_data, test_data, 10, 7, True)
    all_features_prediction(training_data, test_data, 10, 7, False)