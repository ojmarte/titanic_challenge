import pickle
import os

from pipeline import create_preprocessing_pipeline, create_feature_engineering_pipeline, create_ml_pipeline, prepare_submission, create_single_preprocessing_pipeline

os.chdir('../data')


# Tuve que modificar los parametros de entrada para que funcionara create_single_preprocessing_pipeline
train_df = create_preprocessing_pipeline('../data/train.csv', True)
# train_df = create_single_preprocessing_pipeline

# Tuve que modificar los parametros de entrada para que funcionara
features_df = create_feature_engineering_pipeline(train_df)

model, training_acc = create_ml_pipeline(features_df)

print('Model trained successfully, acc: ', training_acc)

pickle.dump(model, open(f'dt_classifier_acc_{round(training_acc)}', 'wb'))

#  Tuve que modificar esta linea de codigo para que funcionara usando el notebook
submission_df = prepare_submission(model, '../data/test.csv', '../data/submission.csv')
passager_row = submission_df.loc[submission_df['PassengerId'] == 894]
print(int(passager_row.iloc[0]['Survived']))

# titanic.loc[titanic["Age"] > 35, "Name"]

# PassengerId  Pclass  Sex  Age  Fare  Embarked  Deck  Title  Relatives  Age_Class