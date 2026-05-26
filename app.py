from flask import Flask, render_template, request, jsonify
import numpy as np
import json
import os
import sys
import traceback
from datetime import datetime
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

class NormalGenerator:
    def __init__(self, mean_func, std):
        self.mean_func = mean_func
        self.std = std

    def __call__(self, parameter):
        mean = self.mean_func(parameter)
        return np.random.normal(mean, self.std)


class UniformGenerator:
    def __init__(self, low_func, high_func):
        self.low_func = low_func
        self.high_func = high_func

    def __call__(self, parameter):
        low = self.low_func(parameter)
        high = self.high_func(parameter)
        return np.random.uniform(low, high)


def Gibbs_sampling(condition_ditribution_list, init_variance, iteration_step, burn_in=0, thinning=1, random_seed=None):
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
            if len(samples) % 5000 == 0:
                pass

    return samples


def compute_autocorrelation(data, max_lag=50):
    n = len(data)
    if n < 2:
        return [1.0]
    
    mean = np.mean(data)
    var = np.var(data)
    
    if var == 0:
        return [1.0] + [0.0] * (min(max_lag, n) - 1)
    
    autocorr = []
    max_lag = min(max_lag, n)
    for lag in range(max_lag):
        if lag == 0:
            autocorr.append(1.0)
        else:
            if n - lag < 1:
                autocorr.append(0.0)
            else:
                cov = np.mean((data[:-lag] - mean) * (data[lag:] - mean))
                autocorr.append(cov / var)
    
    return autocorr


def compute_ess(samples):
    n = len(samples)
    if n < 2:
        return n
    
    acf = compute_autocorrelation(samples, max_lag=50)
    
    sum_autocorr = 1.0
    for lag in range(1, len(acf)):
        if acf[lag] < 0.05:
            break
        sum_autocorr += 2 * acf[lag]
    
    if sum_autocorr <= 0:
        return n
    
    return int(n / sum_autocorr)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})


@app.route('/api/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': '请求体不能为空'}), 400

        param_x = data.get('param_x', [0.5, 0.3, 0.8])
        param_y = data.get('param_y', [0.5, 0.4, 0.7])
        param_z = data.get('param_z', [0.3, 0.4, 0.9])
        init_values = data.get('init_values', [1.0, 2.0, 2.0])
        iterations = data.get('iterations', 5000)
        burn_in = data.get('burn_in', 500)
        thinning = data.get('thinning', 1)
        random_seed = data.get('random_seed', None)
        dist_types = data.get('dist_types', ['normal', 'normal', 'normal'])

        iterations = min(iterations, 15000)
        burn_in = min(burn_in, iterations // 2)
        thinning = max(1, min(thinning, 10))

        params = [
            param_x[:-1],
            param_y[:-1],
            param_z[:-1]
        ]
        stds = [param_x[-1], param_y[-1], param_z[-1]]

        condition_dists = []
        for i, dist_type in enumerate(dist_types):
            if dist_type == 'normal':
                def create_mean(idx=i):
                    return lambda p: sum(params[idx][j] * p[j] for j in range(len(params[idx]))) if len(params[idx]) > 0 else 0.0
                condition_dists.append(NormalGenerator(mean_func=create_mean(), std=stds[i]))
            elif dist_type == 'uniform':
                def create_low(idx=i):
                    return lambda p: sum(params[idx][j] * p[j] for j in range(len(params[idx]))) - stds[i] if len(params[idx]) > 0 else -stds[i]
                def create_high(idx=i):
                    return lambda p: sum(params[idx][j] * p[j] for j in range(len(params[idx]))) + stds[i] if len(params[idx]) > 0 else stds[i]
                condition_dists.append(UniformGenerator(low_func=create_low(), high_func=create_high()))
            else:
                def create_mean(idx=i):
                    return lambda p: sum(params[idx][j] * p[j] for j in range(len(params[idx]))) if len(params[idx]) > 0 else 0.0
                condition_dists.append(NormalGenerator(mean_func=create_mean(), std=stds[i]))

        samples = Gibbs_sampling(condition_dists, init_values, iterations, burn_in, thinning, random_seed)
        
        if len(samples) == 0:
            return jsonify({'error': '未生成任何样本'}), 500

        samples_array = np.array(samples)

        diagnostics = []
        for i in range(samples_array.shape[1]):
            var_data = samples_array[:, i]
            acf = compute_autocorrelation(var_data, max_lag=30)
            diagnostics.append({
                'mean': float(np.mean(var_data)),
                'std': float(np.std(var_data)),
                'var': float(np.var(var_data)),
                'min': float(np.min(var_data)),
                'max': float(np.max(var_data)),
                'median': float(np.median(var_data)),
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
            'samples': samples[:10000],
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

        return jsonify(result)

    except Exception as e:
        error_info = {
            'error': str(e),
            'type': type(e).__name__,
            'timestamp': datetime.now().isoformat()
        }
        print(f"Error in /api/generate: {error_info}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return jsonify(error_info), 500


@app.route('/api/load_json', methods=['GET'])
def load_json():
    try:
        json_path = os.path.join(os.path.dirname(__file__), 'json', 'gibbs_results.json')
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                return jsonify(json.load(f))
        return jsonify({'error': 'No data found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/list_files', methods=['GET'])
def list_files():
    try:
        json_dir = os.path.join(os.path.dirname(__file__), 'json')
        if os.path.exists(json_dir):
            files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
            files.sort(reverse=True)
            return jsonify({'files': files[:20]})
        return jsonify({'files': []})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/load_file/<filename>', methods=['GET'])
def load_file(filename):
    try:
        if '..' in filename or '/' in filename:
            return jsonify({'error': 'Invalid filename'}), 400
        
        json_path = os.path.join(os.path.dirname(__file__), 'json', filename)
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                return jsonify(json.load(f))
        return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/presets', methods=['GET'])
def get_presets():
    presets = {
        'default': {
            'name': '默认三元正态',
            'param_x': [0.5, 0.3, 0.8],
            'param_y': [0.5, 0.4, 0.7],
            'param_z': [0.3, 0.4, 0.9],
            'init_values': [1.0, 2.0, 2.0],
            'iterations': 5000,
            'burn_in': 500,
            'thinning': 1,
            'dist_types': ['normal', 'normal', 'normal']
        },
        'strong_correlation': {
            'name': '强相关性',
            'param_x': [0.8, 0.6, 0.5],
            'param_y': [0.8, 0.6, 0.5],
            'param_z': [0.6, 0.6, 0.5],
            'init_values': [0.0, 0.0, 0.0],
            'iterations': 8000,
            'burn_in': 1000,
            'thinning': 1,
            'dist_types': ['normal', 'normal', 'normal']
        },
        'weak_correlation': {
            'name': '弱相关性',
            'param_x': [0.1, 0.1, 1.0],
            'param_y': [0.1, 0.1, 1.0],
            'param_z': [0.1, 0.1, 1.0],
            'init_values': [0.0, 0.0, 0.0],
            'iterations': 5000,
            'burn_in': 300,
            'thinning': 1,
            'dist_types': ['normal', 'normal', 'normal']
        },
        'mixed_distributions': {
            'name': '混合分布',
            'param_x': [0.5, 0.3, 0.8],
            'param_y': [0.5, 0.4, 1.0],
            'param_z': [0.3, 0.4, 1.0],
            'init_values': [0.0, 0.0, 0.0],
            'iterations': 6000,
            'burn_in': 500,
            'thinning': 1,
            'dist_types': ['normal', 'uniform', 'normal']
        }
    }
    return jsonify(presets)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
