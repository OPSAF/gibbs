from datetime import datetime
import csv
import io
import json
import os
import sys
import traceback

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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 智能选择存储路径：优先 /tmp（云环境可写），其次本地json目录，最后纯内存
def get_storage_dir():
    # 云环境通常有可写的 /tmp 目录
    tmp_dirs = ['/tmp', '/var/tmp', tempfile.gettempdir()] if 'tempfile' in sys.modules else ['/tmp', '/var/tmp']
    for d in tmp_dirs:
        try:
            test_path = os.path.join(d, '.write_test')
            with open(test_path, 'w') as f:
                f.write('test')
            os.remove(test_path)
            return os.path.join(d, 'gibbs_json')
        except (OSError, IOError, PermissionError):
            continue
    # 回退到本地目录（可能不可写）
    local_dir = os.path.join(BASE_DIR, "json")
    try:
        os.makedirs(local_dir, exist_ok=True)
        test_path = os.path.join(local_dir, '.write_test')
        with open(test_path, 'w') as f:
            f.write('test')
        os.remove(test_path)
        return local_dir
    except (OSError, IOError, PermissionError):
        pass
    # 完全只读环境：返回None表示使用内存存储
    return None

# 延迟导入 tempfile
try:
    import tempfile
except ImportError:
    pass

JSON_DIR = get_storage_dir()
IS_READ_ONLY_FS = JSON_DIR is None

# 内存存储后备方案（用于完全只读的云环境）
_in_memory_store = {
    'latest': None,
    'history': []
}

MAX_ITERATIONS = 20000
MAX_RETURNED_SAMPLES = 12000
MAX_MEMORY_HISTORY = 20


DISTRIBUTIONS = {
    "normal": {
        "label": "正态 Normal",
        "kind": "continuous",
        "description": "围绕条件均值对称波动，适合作为默认高斯条件分布。",
    },
    "uniform": {
        "label": "均匀 Uniform",
        "kind": "continuous",
        "description": "在条件均值附近的固定区间内等概率采样。",
    },
    "laplace": {
        "label": "拉普拉斯 Laplace",
        "kind": "continuous",
        "description": "尖峰厚尾，适合模拟更容易出现极端跳动的链。",
    },
    "student_t": {
        "label": "学生 t Student-t",
        "kind": "continuous",
        "description": "厚尾连续分布，对异常值更宽容。",
    },
    "logistic": {
        "label": "逻辑斯蒂 Logistic",
        "kind": "continuous",
        "description": "形状接近正态但尾部稍厚，适合平滑 S 型扰动。",
    },
    "triangular": {
        "label": "三角 Triangular",
        "kind": "continuous",
        "description": "在左右边界内采样，越靠近条件均值概率越高。",
    },
    "exponential": {
        "label": "指数 Exponential",
        "kind": "continuous",
        "description": "右偏分布，适合等待时间、寿命类非对称变量。",
    },
    "gamma": {
        "label": "伽马 Gamma",
        "kind": "continuous",
        "description": "右偏且非负趋势明显，适合强度、尺度、累计量。",
    },
    "beta": {
        "label": "贝塔 Beta",
        "kind": "bounded",
        "description": "在条件均值附近有边界的柔性分布，适合比例类变量。",
    },
    "poisson": {
        "label": "泊松 Poisson",
        "kind": "discrete",
        "description": "离散计数分布，条件均值会被映射为发生率。",
    },
}


def clamp(value, low, high):
    return max(low, min(high, value))


def as_float(value, default=0.0):
    try:
        result = float(value)
        if not np.isfinite(result):
            return default
        return result
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
    merged = values[:length] + default[len(values):]
    return [as_float(v, default[i]) for i, v in enumerate(merged[:length])]


def positive_scale(value):
    return clamp(abs(as_float(value, 1.0)), 1e-6, 1_000.0)


def conditional_location(coefficients, parameters):
    if len(coefficients) == 0:
        return 0.0
    return float(sum(coefficients[j] * parameters[j] for j in range(len(coefficients))))


def sample_distribution(rng, dist_type, location, scale):
    scale = positive_scale(scale)
    dist_type = dist_type if dist_type in DISTRIBUTIONS else "normal"

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
        raw = rng.gamma(shape, scale / shape)
        return location + raw - scale
    if dist_type == "beta":
        raw = rng.beta(2.0, 2.0)
        return location + (raw - 0.5) * 2.0 * scale
    if dist_type == "poisson":
        rate = clamp(np.exp(clamp(location, -8.0, 8.0)), 1e-6, 5000.0)
        return float(rng.poisson(rate))
    return rng.normal(location, scale)


def build_conditionals(params, scales, dist_types):
    conditionals = []
    for index, dist_type in enumerate(dist_types):
        coefficients = params[index]
        scale = scales[index]

        def sampler(other_parameters, coefficients=coefficients, scale=scale, dist_type=dist_type):
            location = conditional_location(coefficients, other_parameters)
            return sample_distribution(np.random.default_rng(), dist_type, location, scale)

        conditionals.append(sampler)
    return conditionals


def gibbs_sampling(params, scales, dist_types, init_values, iterations, burn_in=0, thinning=1, random_seed=None):
    rng = np.random.default_rng(random_seed)
    current_state = np.array(init_values, dtype=float)
    samples = []
    total_steps = burn_in + iterations * thinning

    for step in range(total_steps):
        for index, dist_type in enumerate(dist_types):
            others = np.delete(current_state, index)
            location = conditional_location(params[index], others)
            current_state[index] = sample_distribution(rng, dist_type, location, scales[index])

        if step >= burn_in and (step - burn_in) % thinning == 0:
            samples.append(current_state.copy().tolist())

    return samples


def compute_autocorrelation(data, max_lag=50):
    values = np.asarray(data, dtype=float)
    n = len(values)
    if n < 2:
        return [1.0]

    mean = np.mean(values)
    variance = np.var(values)
    if variance <= 1e-14:
        return [1.0] + [0.0] * (min(max_lag, n) - 1)

    autocorr = []
    max_lag = min(max_lag, n)
    for lag in range(max_lag):
        if lag == 0:
            autocorr.append(1.0)
            continue
        cov = np.mean((values[:-lag] - mean) * (values[lag:] - mean))
        autocorr.append(float(cov / variance))
    return autocorr


def compute_ess(samples):
    n = len(samples)
    if n < 2:
        return n

    acf = compute_autocorrelation(samples, max_lag=80)
    autocorr_sum = 1.0
    for lag in range(1, len(acf)):
        if acf[lag] <= 0.03:
            break
        autocorr_sum += 2.0 * acf[lag]
    return int(n / autocorr_sum) if autocorr_sum > 0 else n


def summarize_samples(samples_array):
    diagnostics = []
    for index in range(samples_array.shape[1]):
        values = samples_array[:, index]
        q05, q25, q75, q95 = np.percentile(values, [5, 25, 75, 95])
        diagnostics.append({
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "var": float(np.var(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
            "q05": float(q05),
            "q25": float(q25),
            "q75": float(q75),
            "q95": float(q95),
            "ess": compute_ess(values),
            "autocorrelation": [float(a) for a in compute_autocorrelation(values, max_lag=40)],
        })
    return diagnostics


def build_insights(samples_array, diagnostics, dist_types):
    corr = np.corrcoef(samples_array, rowvar=False)
    strongest = {"pair": "X-Y", "value": float(corr[0, 1])}
    for pair, value in {"X-Z": corr[0, 2], "Y-Z": corr[1, 2]}.items():
        if abs(value) > abs(strongest["value"]):
            strongest = {"pair": pair, "value": float(value)}

    ess_values = [d["ess"] for d in diagnostics]
    min_ess = min(ess_values)
    quality = "优秀" if min_ess > len(samples_array) * 0.45 else "可用" if min_ess > len(samples_array) * 0.2 else "偏低"
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

    return {
        "quality": quality,
        "strongest_correlation": strongest,
        "suggestions": suggestions,
    }


def normalize_payload(data):
    param_x = vector(data.get("param_x"), [0.5, 0.3, 0.8], 3)
    param_y = vector(data.get("param_y"), [0.5, 0.4, 0.7], 3)
    param_z = vector(data.get("param_z"), [0.3, 0.4, 0.9], 3)
    init_values = vector(data.get("init_values"), [1.0, 2.0, 2.0], 3)

    dist_types = data.get("dist_types", ["normal", "normal", "normal"])
    if not isinstance(dist_types, list):
        dist_types = ["normal", "normal", "normal"]
    dist_types = [(dist if dist in DISTRIBUTIONS else "normal") for dist in (dist_types[:3] + ["normal"] * 3)[:3]]

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


def result_payload(config, samples):
    samples_array = np.array(samples, dtype=float)
    diagnostics = summarize_samples(samples_array)
    stats = {
        "mean": np.mean(samples_array, axis=0).tolist(),
        "std": np.std(samples_array, axis=0).tolist(),
        "var": np.var(samples_array, axis=0).tolist(),
        "min": np.min(samples_array, axis=0).tolist(),
        "max": np.max(samples_array, axis=0).tolist(),
        "q05": np.percentile(samples_array, 5, axis=0).tolist(),
        "q95": np.percentile(samples_array, 95, axis=0).tolist(),
        "covariance_matrix": np.cov(samples_array, rowvar=False).tolist(),
        "correlation_matrix": np.corrcoef(samples_array, rowvar=False).tolist(),
    }
    return {
        "parameters": {**config, "timestamp": datetime.now().isoformat()},
        "samples": samples[:MAX_RETURNED_SAMPLES],
        "statistics": stats,
        "diagnostics": diagnostics,
        "n_samples": len(samples),
        "effective_sample_sizes": [d["ess"] for d in diagnostics],
        "insights": build_insights(samples_array, diagnostics, config["dist_types"]),
        "distributions": DISTRIBUTIONS,
    }


def save_result(result):
    """智能保存：尝试文件系统，失败则存入内存"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"gibbs_results_{timestamp}.json"
    
    # 总是更新内存存储
    global _in_memory_store
    memory_entry = {'filename': filename, 'data': result, 'timestamp': timestamp}
    _in_memory_store['latest'] = result
    _in_memory_store['history'].insert(0, memory_entry)
    if len(_in_memory_store['history']) > MAX_MEMORY_HISTORY:
        _in_memory_store['history'] = _in_memory_store['history'][:MAX_MEMORY_HISTORY]
    
    # 尝试文件系统保存
    if IS_READ_ONLY_FS or JSON_DIR is None:
        return f"{filename} (memory)"
    
    try:
        os.makedirs(JSON_DIR, exist_ok=True)
        filepath = os.path.join(JSON_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
        
        latest_path = os.path.join(JSON_DIR, "gibbs_results.json")
        with open(latest_path, "w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
        
        return filename
    except (OSError, IOError, PermissionError) as e:
        print(f"[Warning] File system write failed ({e}), using memory storage", file=sys.stderr)
        return f"{filename} (memory)"


def load_from_storage(filename='gibbs_results.json'):
    """统一加载：优先文件系统，否则从内存加载"""
    if not IS_READ_ONLY_FS and JSON_DIR is not None:
        filepath = os.path.join(JSON_DIR, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as file:
                    return json.load(file)
            except Exception:
                pass
    
    # 回退到内存
    if filename == 'gibbs_results.json':
        return _in_memory_store.get('latest')
    
    for entry in _in_memory_store.get('history', []):
        if entry.get('filename') == filename:
            return entry.get('data')
    
    return None


def list_storage_files():
    """列出所有可用文件（合并文件系统和内存）"""
    files_set = set()
    
    if not IS_READ_ONLY_FS and JSON_DIR is not None and os.path.exists(JSON_DIR):
        try:
            files_set.update(
                name for name in os.listdir(JSON_DIR) 
                if name.endswith('.json') and name != 'gibbs_results.json'
            )
        except OSError:
            pass
    
    for entry in _in_memory_store.get('history', []):
        fname = entry.get('filename', '')
        if fname.endswith('.json'):
            files_set.add(fname)
    
    return sorted(files_set, reverse=True)[:30]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "storage_mode": "memory" if IS_READ_ONLY_FS else ("filesystem: " + JSON_DIR),
        "timestamp": datetime.now().isoformat()
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

        samples = gibbs_sampling(
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

        result = result_payload(config, samples)
        if data.get("save", True):
            result["saved_file"] = save_result(result)
        return jsonify(result)

    except Exception as exc:
        error_info = {
            "error": str(exc),
            "type": type(exc).__name__,
            "timestamp": datetime.now().isoformat(),
        }
        print(f"Error in /api/generate: {error_info}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return jsonify(error_info), 500


@app.route("/api/export_csv", methods=["POST"])
def export_csv():
    try:
        data = request.get_json() or {}
        samples = data.get("samples", [])
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["iteration", "X", "Y", "Z"])
        for index, row in enumerate(samples):
            if isinstance(row, list) and len(row) >= 3:
                writer.writerow([index, row[0], row[1], row[2]])
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=gibbs_samples.csv"},
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/load_json", methods=["GET"])
def load_json():
    try:
        result = load_from_storage('gibbs_results.json')
        if result:
            return jsonify(result)
        return jsonify({"error": "No data found"}), 404
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/list_files", methods=["GET"])
def list_files():
    try:
        files = list_storage_files()
        return jsonify({"files": files})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/load_file/<filename>", methods=["GET"])
def load_file(filename):
    try:
        if ".." in filename or "/" in filename or "\\" in filename:
            return jsonify({"error": "Invalid filename"}), 400

        result = load_from_storage(filename)
        if result:
            return jsonify(result)
        return jsonify({"error": "File not found"}), 404
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


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
