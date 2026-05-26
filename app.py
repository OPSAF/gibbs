from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import json
import os
import time
from datetime import datetime

app = Flask(__name__)

class NormalGenerator:
    """条件正态分布的随机数生成器"""
    def __init__(self, mean_func, std):
        self.mean_func = mean_func
        self.std = std

    def __call__(self, parameter):
        mean = self.mean_func(parameter)
        return np.random.normal(mean, self.std)


class UniformGenerator:
    """条件均匀分布的随机数生成器"""
    def __init__(self, low_func, high_func):
        self.low_func = low_func
        self.high_func = high_func

    def __call__(self, parameter):
        low = self.low_func(parameter)
        high = self.high_func(parameter)
        return np.random.uniform(low, high)


def Gibbs_sampling(condition_ditribution_list, init_variance, iteration_step, burn_in=0, thinning=1, random_seed=None):
    """Gibbs采样核心算法"""
    if random_seed is not None:
        np.random.seed(random_seed)
    
    current_state = np.array(init_variance, dtype=float)
    samples = []
    total_steps = burn_in + iteration_step * thinning

    for i in range(total_steps):
        for j in range(len(condition_ditribution_list)):
            temp_list = np.delete(current_state, j)
            current_state[j] = condition_ditribution_list[j](temp_list)
        
        if i >= burn_in and (i - burn_in) % thinning == 0:
            samples.append(current_state.copy().tolist())

    return samples


def compute_autocorrelation(data, max_lag=50):
    """计算自相关系数"""
    n = len(data)
    mean = np.mean(data)
    var = np.var(data)
    
    autocorr = []
    for lag in range(min(max_lag, n)):
        if lag == 0:
            autocorr.append(1.0)
        else:
            cov = np.mean((data[:-lag] - mean) * (data[lag:] - mean))
            autocorr.append(cov / var)
    
    return autocorr


def compute_ess(samples):
    """计算有效样本大小 (Effective Sample Size)"""
    n = len(samples)
    acf = compute_autocorrelation(samples, max_lag=50)
    
    # 求和直到自相关小于0.05
    sum_autocorr = 1.0
    for lag in range(1, len(acf)):
        if acf[lag] < 0.05:
            break
        sum_autocorr += 2 * acf[lag]
    
    return int(n / sum_autocorr)


def create_gibbs_sampler(params, dist_types):
    """根据参数创建Gibbs采样器（支持多种分布类型）"""
    generators = []
    
    for i, (param, dist_type) in enumerate(zip(params, dist_types)):
        if dist_type == 'normal':
            def create_mean_func(idx):
                return lambda p: sum(param[j] * p[j] for j in range(len(param)))
            
            gen = NormalGenerator(
                mean_func=create_mean_func(i),
                std=param[-1]
            )
        elif dist_type == 'uniform':
            def create_low_func(idx):
                return lambda p: sum(param[j] * p[j] for j in range(len(param)-2)) - param[-2]
            
            def create_high_func(idx):
                return lambda p: sum(param[j] * p[j] for j in range(len(param)-2)) + param[-2]
            
            gen = UniformGenerator(
                low_func=create_low_func(i),
                high_func=create_high_func(i)
            )
        else:
            gen = NormalGenerator(
                mean_func=lambda p: sum(param[j] * p[j] for j in range(len(param)-1)),
                std=param[-1]
            )
        
        generators.append(gen)
    
    return generators


@app.route('/')
def index():
    """渲染HTML页面"""
    return render_template('index.html')


@app.route('/api/generate', methods=['POST'])
def generate():
    """生成Gibbs采样数据"""
    data = request.get_json()
    
    # 基础参数
    param_x = data.get('param_x', [0.5, 0.3, 0.8])
    param_y = data.get('param_y', [0.5, 0.4, 0.7])
    param_z = data.get('param_z', [0.3, 0.4, 0.9])
    init_values = data.get('init_values', [1.0, 2.0, 2.0])
    iterations = data.get('iterations', 10000)
    burn_in = data.get('burn_in', 1000)
    
    # 高级参数
    thinning = data.get('thinning', 1)
    random_seed = data.get('random_seed', None)
    dist_types = data.get('dist_types', ['normal', 'normal', 'normal'])
    
    # 准备参数（最后一个是标准差）
    params = [
        param_x[:-1],  # 系数部分
        param_y[:-1],
        param_z[:-1]
    ]
    stds = [param_x[-1], param_y[-1], param_z[-1]]
    
    # 创建条件分布生成器
    condition_dists = []
    for i, dist_type in enumerate(dist_types):
        if dist_type == 'normal':
            def create_mean(idx=i):
                return lambda p: sum(params[idx][j] * p[j] for j in range(len(params[idx])))
            condition_dists.append(NormalGenerator(mean_func=create_mean(), std=stds[i]))
        elif dist_type == 'uniform':
            def create_low(idx=i):
                return lambda p: sum(params[idx][j] * p[j] for j in range(len(params[idx]))) - stds[i]
            def create_high(idx=i):
                return lambda p: sum(params[idx][j] * p[j] for j in range(len(params[idx]))) + stds[i]
            condition_dists.append(UniformGenerator(low_func=create_low(), high_func=create_high()))
        else:
            def create_mean(idx=i):
                return lambda p: sum(params[idx][j] * p[j] for j in range(len(params[idx])))
            condition_dists.append(NormalGenerator(mean_func=create_mean(), std=stds[i]))

    samples = Gibbs_sampling(condition_dists, init_values, iterations, burn_in, thinning, random_seed)
    samples_array = np.array(samples)

    # 计算统计量
    diagnostics = []
    for i in range(samples_array.shape[1]):
        var_data = samples_array[:, i]
        acf = compute_autocorrelation(var_data, max_lag=50)
        diagnostics.append({
            'mean': float(np.mean(var_data)),
            'std': float(np.std(var_data)),
            'var': float(np.var(var_data)),
            'min': float(np.min(var_data)),
            'max': float(np.max(var_data)),
            'median': float(np.median(var_data)),
            'skewness': float((np.mean((var_data - np.mean(var_data))**3)) / (np.std(var_data)**3)),
            'kurtosis': float((np.mean((var_data - np.mean(var_data))**4)) / (np.std(var_data)**4) - 3),
            'ess': compute_ess(var_data),
            'autocorrelation': [float(a) for a in acf]
        })

    result = {
        'parameters': {
            'param_x': param_x,
            'param_y': param_y,
            'param_z': param_z,
            'init_values': init_values,
            'iterations': iterations,
            'burn_in': burn_in,
            'thinning': thinning,
            'random_seed': random_seed,
            'dist_types': dist_types,
            'timestamp': datetime.now().isoformat()
        },
        'samples': samples,
        'statistics': {
            'mean': np.mean(samples_array, axis=0).tolist(),
            'std': np.std(samples_array, axis=0).tolist(),
            'var': np.var(samples_array, axis=0).tolist(),
            'min': np.min(samples_array, axis=0).tolist(),
            'max': np.max(samples_array, axis=0).tolist(),
            'covariance_matrix': np.cov(samples_array, rowvar=False).tolist(),
            'correlation_matrix': np.corrcoef(samples_array, rowvar=False).tolist()
        },
        'diagnostics': diagnostics,
        'n_samples': len(samples),
        'effective_sample_sizes': [d['ess'] for d in diagnostics]
    }

    json_dir = os.path.join(os.path.dirname(__file__), 'json')
    os.makedirs(json_dir, exist_ok=True)
    
    # 保存带时间戳的文件
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(os.path.join(json_dir, f'gibbs_results_{timestamp}.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    # 同时保存为最新文件
    with open(os.path.join(json_dir, 'gibbs_results.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return jsonify(result)


@app.route('/api/load_json', methods=['GET'])
def load_json():
    """加载JSON文件数据"""
    json_path = os.path.join(os.path.dirname(__file__), 'json', 'gibbs_results.json')
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            return jsonify(json.load(f))
    return jsonify({'error': 'No data found'}), 404


@app.route('/api/list_files', methods=['GET'])
def list_files():
    """列出所有已保存的JSON文件"""
    json_dir = os.path.join(os.path.dirname(__file__), 'json')
    if os.path.exists(json_dir):
        files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        files.sort(reverse=True)
        return jsonify({'files': files})
    return jsonify({'files': []})


@app.route('/api/load_file/<filename>', methods=['GET'])
def load_file(filename):
    """加载指定的JSON文件"""
    json_path = os.path.join(os.path.dirname(__file__), 'json', filename)
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            return jsonify(json.load(f))
    return jsonify({'error': 'File not found'}), 404


@app.route('/api/export', methods=['POST'])
def export_data():
    """导出数据为CSV"""
    data = request.get_json()
    samples = np.array(data.get('samples', []))
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(os.path.dirname(__file__), 'json', f'samples_{timestamp}.csv')
    
    header = ','.join([f'Var_{i+1}' for i in range(samples.shape[1])])
    np.savetxt(csv_path, samples, delimiter=',', header=header, comments='')
    
    return jsonify({'message': 'CSV exported', 'filename': f'samples_{timestamp}.csv'})


@app.route('/api/presets', methods=['GET'])
def get_presets():
    """获取预设配置"""
    presets = {
        'default': {
            'name': '默认三元正态',
            'param_x': [0.5, 0.3, 0.8],
            'param_y': [0.5, 0.4, 0.7],
            'param_z': [0.3, 0.4, 0.9],
            'init_values': [1.0, 2.0, 2.0],
            'iterations': 10000,
            'burn_in': 1000,
            'thinning': 1,
            'dist_types': ['normal', 'normal', 'normal']
        },
        'strong_correlation': {
            'name': '强相关性',
            'param_x': [0.8, 0.6, 0.5],
            'param_y': [0.8, 0.6, 0.5],
            'param_z': [0.6, 0.6, 0.5],
            'init_values': [0.0, 0.0, 0.0],
            'iterations': 20000,
            'burn_in': 2000,
            'thinning': 1,
            'dist_types': ['normal', 'normal', 'normal']
        },
        'weak_correlation': {
            'name': '弱相关性',
            'param_x': [0.1, 0.1, 1.0],
            'param_y': [0.1, 0.1, 1.0],
            'param_z': [0.1, 0.1, 1.0],
            'init_values': [0.0, 0.0, 0.0],
            'iterations': 10000,
            'burn_in': 500,
            'thinning': 1,
            'dist_types': ['normal', 'normal', 'normal']
        },
        'mixed_distributions': {
            'name': '混合分布',
            'param_x': [0.5, 0.3, 0.8],
            'param_y': [0.5, 0.4, 1.0],
            'param_z': [0.3, 0.4, 1.0],
            'init_values': [0.0, 0.0, 0.0],
            'iterations': 15000,
            'burn_in': 1000,
            'thinning': 1,
            'dist_types': ['normal', 'uniform', 'normal']
        }
    }
    return jsonify(presets)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
