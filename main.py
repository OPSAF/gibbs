import numpy as np
import json
import os

class NormalGenerator:
    """条件正态分布的随机数生成器"""

    def __init__(self, mean_func, std):
        """
        :param mean_func: 条件均值函数,输入为其他变量的值列表
        :param std: 条件标准差
        """
        self.mean_func = mean_func
        self.std = std

    def __call__(self, parameter):
        """
        :param parameter: 其他变量的当前值列表
        :return: 从条件分布中采样的随机数
        """
        mean = self.mean_func(parameter)
        return np.random.normal(mean, self.std)


def Gibbs_sampling(condition_ditribution_list, init_variance, iteration_step, burn_in=0):
    '''
    :param condition_ditribution_list: 条件分布的随机数生成器列表
    :param init_variance: 初始参数值
    :param iteration_step: 迭代步数
    :param burn_in: 预烧期步数,丢弃前burn_in个样本
    :return: 采样得到的样本列表
    '''
    assert (len(condition_ditribution_list) == len(init_variance))

    current_state = init_variance.copy()
    samples = []

    total_steps = burn_in + iteration_step

    for i in range(total_steps):
        for j in range(len(condition_ditribution_list)):
            temp_list = [current_state[k] for k in range(len(current_state)) if k != j]
            current_state[j] = condition_ditribution_list[j](temp_list)

        if i >= burn_in:
            samples.append(current_state.copy())

    return samples


def export_samples_to_json(samples, file_path):
    """
    将采样结果导出为JSON文件

    :param samples: 采样得到的样本列表
    :param file_path: 保存JSON文件的路径
    """
    samples_array = np.array(samples)
    result = {
        'samples': samples_array.tolist(),
        'n_samples': len(samples),
        'n_variables': len(samples[0]) if samples else 0,
        'mean': np.mean(samples_array, axis=0).tolist() if samples else [],
        'std': np.std(samples_array, axis=0).tolist() if samples else []
    }
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"采样结果已保存到: {file_path}")


def export_diagnostics_to_json(samples, var_names=None, window_size=50, file_path='diagnostics.json'):
    """
    导出收敛诊断指标到JSON文件

    :param samples: 采样样本数组
    :param var_names: 变量名称列表
    :param window_size: 滚动窗口大小
    :param file_path: 保存JSON文件的路径
    """
    samples = np.array(samples)
    n_samples, n_vars = samples.shape

    if var_names is None:
        var_names = [f'Variable_{i + 1}' for i in range(n_vars)]

    diagnostics = {}
    for var_idx in range(n_vars):
        data = samples[:, var_idx]
        var_name = var_names[var_idx]

        rolling_mean = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
        rolling_var = np.array([np.var(data[i:i + window_size]) for i in range(len(data) - window_size + 1)])

        diagnostics[var_name] = {
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'var': float(np.var(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'final_rolling_mean': float(rolling_mean[-1]) if len(rolling_mean) > 0 else None,
            'final_rolling_var': float(rolling_var[-1]) if len(rolling_var) > 0 else None
        }

    result = {
        'n_samples': n_samples,
        'n_variables': n_vars,
        'window_size': window_size,
        'diagnostics': diagnostics
    }

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"诊断结果已保存到: {file_path}")


def create_ternary_normal_gibbs():
    """
    创建三元正态分布的吉布斯采样器

    条件分布:
    X|Y,Z ~ N(0.5*Y + 0.3*Z, 0.8)
    Y|X,Z ~ N(0.5*X + 0.4*Z, 0.7)
    Z|X,Y ~ N(0.3*X + 0.4*Y, 0.9)
    """

    gen_x = NormalGenerator(
        mean_func=lambda params: 0.5 * params[0] + 0.3 * params[1],
        std=0.8
    )

    gen_y = NormalGenerator(
        mean_func=lambda params: 0.5 * params[0] + 0.4 * params[1],
        std=0.7
    )

    gen_z = NormalGenerator(
        mean_func=lambda params: 0.3 * params[0] + 0.4 * params[1],
        std=0.9
    )

    condition_distribution_list = [gen_x, gen_y, gen_z]

    return condition_distribution_list


if __name__ == "__main__":
    print("="*60)
    print("        Gibbs采样算法实现与应用")
    print("="*60)

    json_dir = os.path.join(os.path.dirname(__file__), 'json')
    os.makedirs(json_dir, exist_ok=True)

    condition_distributions = create_ternary_normal_gibbs()

    init_values = [1.0, 2.0, 2.0]
    burn_in = 10000
    total_iterations = 20000

    print(f"\n参数设置:")
    print(f"  总迭代次数: {total_iterations}")
    print(f"  Burn-in期: {burn_in}")
    print(f"  变量数: {len(condition_distributions)}")

    print("\n正在执行Gibbs采样...")
    samples = Gibbs_sampling(
        condition_ditribution_list=condition_distributions,
        init_variance=init_values,
        iteration_step=total_iterations
    )

    samples_array = np.array(samples)
    effective_samples = samples_array[burn_in:]

    print(f"采样完成!")
    print(f"有效样本数: {len(effective_samples)}")

    print(f"\n样本统计量:")
    print(f"均值: {np.mean(effective_samples, axis=0)}")
    print(f"标准差: {np.std(effective_samples, axis=0)}")

    print(f"\n前10个样本:")
    for i in range(10):
        print(f"Sample {i + 1}: X={effective_samples[i, 0]:.4f}, "
              f"Y={effective_samples[i, 1]:.4f}, "
              f"Z={effective_samples[i, 2]:.4f}")

    result = {
        'total_iterations': total_iterations,
        'burn_in': burn_in,
        'effective_samples': len(effective_samples),
        'statistics': {
            'mean': np.mean(effective_samples, axis=0).tolist(),
            'std': np.std(effective_samples, axis=0).tolist()
        },
        'first_10_samples': [
            {'X': float(effective_samples[i, 0]), 'Y': float(effective_samples[i, 1]), 'Z': float(effective_samples[i, 2])}
            for i in range(min(10, len(effective_samples)))
        ]
    }

    results_path = os.path.join(json_dir, 'gibbs_sampling_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {results_path}")

    samples_path = os.path.join(json_dir, 'samples.json')
    export_samples_to_json(effective_samples.tolist(), samples_path)

    diagnostics_path = os.path.join(json_dir, 'diagnostics.json')
    export_diagnostics_to_json(effective_samples, var_names=['X', 'Y', 'Z'], file_path=diagnostics_path)

    print("\n" + "="*60)
    print("                    执行完成")
    print("="*60)
    print(f"所有结果已保存到: {json_dir}")
