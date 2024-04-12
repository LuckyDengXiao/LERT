import json
import random

INPUT_PROMPT = '''Determine if the provided function contains any defects. A defect is any kind of issue that would prevent the function from compiling successfully, running correctly, executing efficiently, or adhering to good coding practices.
**Defective Example:**
- **Input:**
```
int divide(int numerator, int denominator) {{
    if (denominator == 0) {{
        throw "Division by zero condition!";
    }}
    return numerator / denominator;
}}
```
- **Output:**
The function is defective.

**Non-Defective Example:**
- **Input:**
```
int add(int a, int b) {{
    return a + b;
}}
```
- **Output:**
The function is non-defective.

**Your Function for Analysis:**

- **Input:**
```
{function}
```
- **Output:**
'''


def build_dataset(data_paths, save_path):
    merged_data = []
    
    for data_path in data_paths:
        with open(data_path, 'r') as f:
            data = json.load(f)
            for sample in data:
                _input = INPUT_PROMPT.format(function=sample['func'])
                _output = 'The function is defective.' if sample['target']==1 else 'The function is non-defective.'
                new_sample = {
                    'conversations': [
                        {'from': 'human', 'value': _input},
                        {'from': 'assistant', 'value': _output}
                    ],
                    'sample': sample
                }
                merged_data.append(new_sample)
    
    # 随机打乱数据集
    random.shuffle(merged_data)
    
    with open(save_path, 'w') as f:
        json.dump(merged_data, f, indent=4, ensure_ascii=False)
    
if __name__ == '__main__':
    # data_paths = ['/shd/zzr/CWE-Predict/dataset/dataset_train.json']
    # save_path = '/shd/zzr/CWE-Predict/dataset/baseline_train.json'
    # build_dataset(data_paths, save_path)
    
    # data_paths = ['/shd/zzr/CWE-Predict/dataset/dataset_train.json', '/shd/zzr/CWE-Predict/dataset/bigvul_generatedData_write.json']
    # save_path = '/shd/zzr/CWE-Predict/dataset/bigvul_train.json'
    # build_dataset(data_paths, save_path)
    
    # data_paths = ['/shd/zzr/CWE-Predict/dataset/dataset_train.json', '/shd/zzr/CWE-Predict/dataset/juliet.json']
    # save_path = '/shd/zzr/CWE-Predict/dataset/juliet_train.json'
    # build_dataset(data_paths, save_path)
    
    data_paths = ['/shd/zzr/CWE-Predict/dataset/dataset_train.json', '/shd/zzr/CWE-Predict/dataset/vgx.json']
    save_path = '/shd/zzr/CWE-Predict/dataset/vgx_train.json'
    build_dataset(data_paths, save_path)
    
    # data_paths = ['/shd/zzr/CWE-Predict/dataset/dataset_test.json']
    # save_path = '/shd/zzr/CWE-Predict/dataset/test.json'
    # build_dataset(data_paths, save_path)
    
    