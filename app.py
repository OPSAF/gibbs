import datetime
import csv
import io
import json
import os
import sys
import threading
import time

from flask import Flask, jsonify, render_template, request, Response
try:
    from flask_cors import CORS
except ImportError:
    def CORS(_app):
        return _app
import numpy as np


app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# 全局配置
MAX_ITERATIONS = 20000
MAX_RETURNED_SAMPLES = 12000
MAX_MEMORY_HISTORY = 15

# 使用纯内存存储（禁用文件系统操作以提高性能）
# 文件系统操作会导致延迟和卡顿，特别是在云环境中
_in_memory_store = {
    'latest': None,
    'history': []
}

# 缓存锁
_store_lock = threading.Lock()


DISTRIBUTIONS = {
    "normal": {"label": "正态 Normal", "kind": "continuous"},
    "uniform": {"label": "均匀 Uniform", "kind": "continuous"},
    "laplace": {"label": "拉普拉斯 Laplace", "kind": "continuous"},
    "student_t": {"label": "学生 t Student-t", "kind": "continuous"},
    "logistic": {"label": "逻辑斯蒂 Logistic", "kind": "continuous"},
    "triangular": {"label": "三角 Triangular", "kind": "continuous"},
    "exponential": {"label": "指数 Exponential", "kind": "continuous"},
    "gamma": {"label": "伽马 Gamma", "kind": "continuous"},
    "beta": {"label": "贝塔 Beta", "kind": "bounded"},
    "poisson": {"label": "泊松 Poisson", "kind": "discrete"},
}


def clamp(value, low, high):
    return max(low, min(high, value))


def as_float(value, default=0.0):
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def as_int(value, default, low=None, high=None):
    try:
        result = int(value)
    except (TypeError, ValueError):
        result = default
    if low is not None:
        result = max(low, result)
    if high is not None:
        result = min(high, result)
    return result


def vector(data, default, length):
    values = list(data) if isinstance(data, (list, tuple)) else []
    return [as_float(values[i], default[i]) for i in range(length)] if len(values) >= length else \
           [as_float(values[i], default[i]) for i in range(len(values))] + default[len(values):]


def positive_scale(value):
    return clamp(abs(as_float(value, 1.0)), 1e-6, 1000.0)


def sample_distribution(rng, dist_type, location, scale):
    scale = positive_scale(scale)
    if dist_type == "normal":
        return rng.normal(location, scale)
    if dist_type == "uniform":
        return rng.uniform(location - scale, location + scale)
    if dist_type == "laplace":
        return rng.laplace(location, scale / np.sqrt(2.0))
    if dist_type == "student_t":
        df = clamp(2.0 + scale * 4.0, 2.2, 30.0)
        return location + rng.standard_t(df) * scale
    if dist_type == "logistic":
        return rng.logistic(location, scale)
    if dist_type == "triangular":
        return rng.triangular(location - scale, location, location + scale)
    if dist_type == "exponential":
        return location + rng.exponential(scale) - scale
    if dist_type == "gamma":
        shape = clamp(2.0 + abs(location) / max(scale, 1e-6), 0.4, 20.0)
        return location + rng.gamma(shape, scale / shape) - scale
    if dist_type == "beta":
        return location + (rng.beta(2.0, 2.0) - 0.5) * 2.0 * scale
    if dist_type == "poisson":
        rate = clamp(np.exp(clamp(location, -8.0, 8.0)), 1e-6, 5000.0)
        return float(rng.poisson(rate))
    return rng.normal(location, scale)


def gibbs_sampling_fast(params, scales, dist_types, init_values, iterations, burn_in=0, thinning=1, random_seed=None):
    """优化版 Gibbs 采样 - 使用 NumPy 向量化和预分配数组"""
    rng = np.random.default_rng(random_seed)
    
    # 预分配结果数组
    n_samples = ((iterations - burn_in) + thinning - 1) // thinning
    samples = np.zeros((n_samples, 3), dtype=np.float64)
    
    current_state = np.array(init_values, dtype=np.float64)
    sample_idx = 0
    
    # 预计算参数
    param_x, param_y, param_z = params
    scale_x, scale_y, scale_z = scales
    dist_x, dist_y, dist_z = dist_types
    
    # 优化后的采样循环
    for step in range(iterations):
        # X | Y, Z
        loc_x = param_x[0] * current_state[1] + param_x[1] * current_state[2]
        current_state[0] = sample_distribution(rng, dist_x, loc_x, scale_x)
        
        # Y | X, Z
        loc_y = param_y[0] * current_state[0] + param_y[1] * current_state[2]
        current_state[1] = sample_distribution(rng, dist_y, loc_y, scale_y)
        
        # Z | X, Y
        loc_z = param_z[0] * current_state[0] + param_z[1] * current_state[1]
        current_state[2] = sample_distribution(rng, dist_z, loc_z, scale_z)
        
        # 存储样本（跳过 burn-in，应用 thinning）
        if step >= burn_in and (step - burn_in) % thinning == 0:
            samples[sample_idx] = current_state
            sample_idx += 1
    
    return samples.tolist()


def compute_autocorrelation_fast(data, max_lag=50):
    """优化版自相关计算"""
    values = np.asarray(data, dtype=np.float64)
    n = len(values)
    if n < 2:
        return [1.0]
    
    # 处理极端值
    finite_mask = np.isfinite(values)
    if np.sum(finite_mask) < 2:
        return [1.0]
    
    median_val = np.median(values[finite_mask])
    values = np.where(finite_mask, values, median_val)
    
    mean = np.mean(values)
    variance = np.var(values)
    if variance <= 1e-14:
        return [1.0] + [0.0] * (min(max_lag, n) - 1)
    
    max_lag = min(max_lag, n)
    autocorr = np.zeros(max_lag)
    autocorr[0] = 1.0
    
    for lag in range(1, max_lag):
        cov = np.mean((values[:-lag] - mean) * (values[lag:] - mean))
        autocorr[lag] = clamp(cov / variance, -1.0, 1.0)
    
    return autocorr.tolist()


def compute_ess_fast(samples):
    """优化版 ESS 计算"""
    n = len(samples)
    if n < 2:
        return n
    
    acf = compute_autocorrelation_fast(samples, max_lag=80)
    autocorr_sum = 1.0
    for lag in range(1, len(acf)):
        if acf[lag] <= 0.03:
            break
        autocorr_sum += 2.0 * max(0.0, acf[lag])
    
    return int(n / autocorr_sum) if autocorr_sum > 0 else n


def summarize_samples_fast(samples_array):
    """优化版样本统计"""
    samples_array = np.asarray(samples_array, dtype=np.float64)
    samples_array = np.nan_to_num(samples_array, nan=0.0, posinf=1e6, neginf=-1e6)
    
    diagnostics = []
    for idx in range(samples_array.shape[1]):
        values = samples_array[:, idx]
        finite_vals = values[np.isfinite(values)]
        
        if len(finite_vals) < 2:
            diagnostics.append({
                "mean": 0.0, "std": 0.0, "var": 0.0, "min": 0.0, "max": 0.0,
                "median": 0.0, "q05": 0.0, "q25": 0.0, "q75": 0.0, "q95": 0.0,
                "ess": 1, "autocorrelation": [1.0]
            })
            continue
        
        q05, q25, q75, q95 = np.percentile(finite_vals, [5, 25, 75, 95])
        diagnostics.append({
            "mean": float(np.mean(finite_vals)),
            "std": float(np.std(finite_vals)),
            "var": float(np.var(finite_vals)),
            "min": float(np.min(finite_vals)),
            "max": float(np.max(finite_vals)),
            "median": float(np.median(finite_vals)),
            "q05": float(q05),
            "q25": float(q25),
            "q75": float(q75),
            "q95": float(q95),
            "ess": compute_ess_fast(finite_vals.tolist()),
            "autocorrelation": [float(a) for a in compute_autocorrelation_fast(finite_vals.tolist(), max_lag=40)]
        })
    return diagnostics


def safe_corrcoef_fast(matrix):
    """优化版相关系数计算"""
    matrix = np.asarray(matrix, dtype=np.float64)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=1e6, neginf=-1e6)
    
    col_means = np.mean(matrix, axis=0)
    col_stds = np.std(matrix, axis=0)
    col_stds[col_stds < 1e-10] = 1.0
    
    normalized = (matrix - col_means) / col_stds
    normalized = np.clip(normalized, -100, 100)
    
    try:
        with np.errstate(all='ignore'):
            corr = np.corrcoef(normalized, rowvar=False)
            corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)
            corr = np.clip(corr, -1.0, 1.0)
            return corr
    except:
        return np.eye(matrix.shape[1])


def build_insights_fast(samples_array, diagnostics, dist_types):
    """优化版洞察生成"""
    corr = safe_corrcoef_fast(samples_array)
    
    strongest = {"pair": "X-Y", "value": float(corr[0, 1])}
    for pair, value in {"X-Z": corr[0, 2], "Y-Z": corr[1, 2]}.items():
        val = float(value)
        if abs(val) > abs(strongest["value"]):
            strongest = {"pair": pair, "value": val}
    
    ess_values = [d["ess"] for d in diagnostics]
    min_ess = min(ess_values) if ess_values else 1
    n_total = len(samples_array)
    quality = "优秀" if min_ess > n_total * 0.45 else "可用" if min_ess > n_total * 0.2 else "偏低"
    
    suggestions = []
    if quality == "偏低":
        suggestions.append("ESS 偏低，建议增加 thinning、降低条件系数或延长迭代。")
    if any(dist in {"student_t", "laplace"} for dist in dist_types):
        suggestions.append("当前包含厚尾分布，极端值更多；观察 5%-95% 区间比只看均值更稳。")
    if any(dist in {"poisson"} for dist in dist_types):
        suggestions.append("泊松变量是离散计数，散点图会呈现阶梯状是正常现象。")
    if abs(strongest["value"]) > 0.75:
        suggestions.append(f"{strongest['pair']} 相关性较强，链可能沿狭长方向移动。")
    if not suggestions:
        suggestions.append("链的基础诊断稳定，可以尝试混合不同分布观察形态变化。")
    
    return {"quality": quality, "strongest_correlation": strongest, "suggestions": suggestions}


def normalize_payload(data):
    """规范化请求参数"""
    param_x = vector(data.get("param_x"), [0.5, 0.3, 0.8], 3)
    param_y = vector(data.get("param_y"), [0.5, 0.4, 0.7], 3)
    param_z = vector(data.get("param_z"), [0.3, 0.4, 0.9], 3)
    init_values = vector(data.get("init_values"), [1.0, 2.0, 2.0], 3)
    
    dist_types = data.get("dist_types", ["normal", "normal", "normal"])
    if not isinstance(dist_types, list):
        dist_types = ["normal", "normal", "normal"]
    dist_types = [(d if d in DISTRIBUTIONS else "normal") for d in dist_types[:3]]
    
    return {
        "param_x": param_x,
        "param_y": param_y,
        "param_z": param_z,
        "init_values": init_values,
        "iterations": as_int(data.get("iterations"), 5000, 100, MAX_ITERATIONS),
        "burn_in": as_int(data.get("burn_in"), 500, 0, MAX_ITERATIONS),
        "thinning": as_int(data.get("thinning"), 1, 1, 20),
        "random_seed": None if data.get("random_seed") in ("", None) else as_int(data.get("random_seed"), 0),
        "dist_types": dist_types,
    }


def result_payload_fast(config, samples):
    """快速生成结果 payload"""
    samples_array = np.asarray(samples, dtype=np.float64)
    samples_array = np.nan_to_num(samples_array, nan=0.0, posinf=1e6, neginf=-1e6)
    
    diagnostics = summarize_samples_fast(samples_array)
    corr_matrix = safe_corrcoef_fast(samples_array)
    
    stats = {
        "mean": np.mean(samples_array, axis=0).tolist(),
        "std": np.std(samples_array, axis=0).tolist(),
        "var": np.var(samples_array, axis=0).tolist(),
        "min": np.min(samples_array, axis=0).tolist(),
        "max": np.max(samples_array, axis=0).tolist(),
        "q05": np.percentile(samples_array, 5, axis=0).tolist(),
        "q95": np.percentile(samples_array, 95, axis=0).tolist(),
        "covariance_matrix": corr_matrix.tolist(),
        "correlation_matrix": corr_matrix.tolist(),
    }
    
    return {
        "parameters": {**config, "timestamp": datetime.datetime.now().isoformat()},
        "samples": samples[:MAX_RETURNED_SAMPLES],
        "statistics": stats,
        "diagnostics": diagnostics,
        "n_samples": len(samples),
        "effective_sample_sizes": [d["ess"] for d in diagnostics],
        "insights": build_insights_fast(samples_array, diagnostics, config["dist_types"]),
        "distributions": DISTRIBUTIONS,
        "saved_file": "memory"
    }


def save_result_fast(result):
    """快速保存结果到内存（无文件操作）"""
    with _store_lock:
        _in_memory_store['latest'] = result
        memory_entry = {
            'filename': f"gibbs_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            'data': result,
            'timestamp': datetime.datetime.now().isoformat()
        }
        _in_memory_store['history'].insert(0, memory_entry)
        if len(_in_memory_store['history']) > MAX_MEMORY_HISTORY:
            _in_memory_store['history'] = _in_memory_store['history'][:MAX_MEMORY_HISTORY]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/docs")
def docs():
    return render_template("docs.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "storage_mode": "memory",
        "timestamp": datetime.datetime.now().isoformat(),
        "history_count": len(_in_memory_store['history'])
    })


@app.route("/api/distributions", methods=["GET"])
def distributions():
    return jsonify(DISTRIBUTIONS)


@app.route("/api/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "请求体不能为空"}), 400
        
        config = normalize_payload(data)
        config["burn_in"] = min(config["burn_in"], config["iterations"])
        
        params = [config["param_x"][:-1], config["param_y"][:-1], config["param_z"][:-1]]
        scales = [config["param_x"][-1], config["param_y"][-1], config["param_z"][-1]]
        
        # 优化：直接调用快速采样函数
        samples = gibbs_sampling_fast(
            params=params,
            scales=scales,
            dist_types=config["dist_types"],
            init_values=config["init_values"],
            iterations=config["iterations"],
            burn_in=config["burn_in"],
            thinning=config["thinning"],
            random_seed=config["random_seed"],
        )
        
        if not samples:
            return jsonify({"error": "未生成任何样本"}), 500
        
        result = result_payload_fast(config, samples)
        if data.get("save", True):
            save_result_fast(result)
        
        return jsonify(result)
    
    except Exception as exc:
        error_info = {
            "error": str(exc),
            "type": type(exc).__name__,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        print(f"Error in /api/generate: {error_info}", file=sys.stderr)
        return jsonify(error_info), 500


@app.route("/api/export_csv", methods=["POST"])
def export_csv():
    try:
        data = request.get_json() or {}
        samples = data.get("samples", [])
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["iteration", "X", "Y", "Z"])
        for idx, row in enumerate(samples[:10000]):
            if isinstance(row, list) and len(row) >= 3:
                writer.writerow([idx, row[0], row[1], row[2]])
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=gibbs_samples.csv"},
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/load_json", methods=["GET"])
def load_json():
    with _store_lock:
        result = _in_memory_store.get('latest')
        if result:
            return jsonify(result)
        return jsonify({"error": "No data found"}), 404


@app.route("/api/list_files", methods=["GET"])
def list_files():
    with _store_lock:
        files = [entry['filename'] for entry in _in_memory_store.get('history', [])]
        return jsonify({"files": files})


@app.route("/api/load_file/<filename>", methods=["GET"])
def load_file(filename):
    if ".." in filename or "/" in filename or "\\" in filename:
        return jsonify({"error": "Invalid filename"}), 400
    
    with _store_lock:
        for entry in _in_memory_store.get('history', []):
            if entry.get('filename') == filename:
                return jsonify(entry.get('data'))
    
    return jsonify({"error": "File not found"}), 404


@app.route("/api/presets", methods=["GET"])
def get_presets():
    presets = {
        "default": {
            "name": "三元高斯基线",
            "note": "稳定、平滑，适合第一次运行。",
            "param_x": [0.5, 0.3, 0.8],
            "param_y": [0.5, 0.4, 0.7],
            "param_z": [0.3, 0.4, 0.9],
            "init_values": [1.0, 2.0, 2.0],
            "iterations": 5000,
            "burn_in": 500,
            "thinning": 1,
            "random_seed": 42,
            "dist_types": ["normal", "normal", "normal"],
        },
        "heavy_tail": {
            "name": "厚尾鲁棒链",
            "note": "Student-t + Laplace 更容易看到极端跳动。",
            "param_x": [0.35, 0.15, 0.9],
            "param_y": [0.45, 0.25, 0.75],
            "param_z": [0.20, 0.35, 0.85],
            "init_values": [0.0, 0.0, 0.0],
            "iterations": 8000,
            "burn_in": 900,
            "thinning": 1,
            "random_seed": 7,
            "dist_types": ["student_t", "laplace", "normal"],
        },
        "bounded_mix": {
            "name": "有界比例混合",
            "note": "Beta 与 Triangular 适合观察边界内波动。",
            "param_x": [0.15, 0.20, 1.0],
            "param_y": [0.35, 0.15, 0.8],
            "param_z": [0.25, 0.25, 0.7],
            "init_values": [0.2, 0.4, 0.6],
            "iterations": 6500,
            "burn_in": 700,
            "thinning": 1,
            "random_seed": 19,
            "dist_types": ["beta", "triangular", "uniform"],
        },
        "skewed_process": {
            "name": "偏态过程",
            "note": "指数与伽马用于模拟等待时间、强度和累计量。",
            "param_x": [0.25, 0.10, 0.7],
            "param_y": [0.20, 0.30, 0.9],
            "param_z": [0.10, 0.15, 0.8],
            "init_values": [1.0, 1.0, 1.0],
            "iterations": 7000,
            "burn_in": 800,
            "thinning": 1,
            "random_seed": 31,
            "dist_types": ["exponential", "gamma", "logistic"],
        },
        "count_hybrid": {
            "name": "计数混合链",
            "note": "包含泊松离散变量，适合计数场景。",
            "param_x": [0.08, 0.04, 0.6],
            "param_y": [0.25, 0.15, 0.7],
            "param_z": [0.15, 0.20, 0.8],
            "init_values": [1.0, 0.0, 0.0],
            "iterations": 6000,
            "burn_in": 600,
            "thinning": 1,
            "random_seed": 11,
            "dist_types": ["poisson", "normal", "laplace"],
        },
        "strong_correlation": {
            "name": "强相关压力测试",
            "note": "高条件系数会让链更慢混合，用于观察 ESS 下降。",
            "param_x": [0.8, 0.6, 0.5],
            "param_y": [0.8, 0.6, 0.5],
            "param_z": [0.6, 0.6, 0.5],
            "init_values": [0.0, 0.0, 0.0],
            "iterations": 9000,
            "burn_in": 1200,
            "thinning": 2,
            "random_seed": 5,
            "dist_types": ["normal", "student_t", "normal"],
        },
    }
    return jsonify(presets)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
