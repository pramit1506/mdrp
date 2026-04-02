import importlib

files = ['api','ensemble_model','evaluate_models','feature_engineering','input_mapper','predict','preprocess','train_models']
for f in files:
    try:
        importlib.import_module(f)
        print('ok', f)
    except Exception as e:
        print('err', f, e)
