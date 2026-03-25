import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from preprocess import TextPreprocessor
    print('✅ preprocess.py imports successfully')
    
    preprocessor = TextPreprocessor()
    test_text = 'I am very happy today!'
    result = preprocessor.clean_text(test_text)
    print(f'✅ Text preprocessing works: "{test_text}" -> "{result}"')
except Exception as e:
    print(f'❌ preprocess.py error: {e}')
    import traceback
    traceback.print_exc()

try:
    import pickle
    import os
    
    # Check if model files exist
    model_files = [
        'core_6_emotions_model.pkl',
        'core_6_emotions_vectorizer.pkl', 
        'core_6_emotions_encoder.pkl',
        'core_6_emotions_mappings.pkl'
    ]
    
    for file in model_files:
        if os.path.exists(file):
            print(f'✅ {file} exists')
        else:
            print(f'❌ {file} missing')
            
except Exception as e:
    print(f'❌ File check error: {e}')

# Test loading the model
try:
    import pickle
    if os.path.exists('core_6_emotions_model.pkl'):
        with open('core_6_emotions_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print('✅ Model loads successfully')
        print(f'Model type: {type(model)}')
    else:
        print('❌ Model file not found')
except Exception as e:
    print(f'❌ Model loading error: {e}')
    import traceback
    traceback.print_exc()
