import os
import re
import subprocess
import time
import datetime
import toml

from helper import utils
from helper import consts

def set_pkt_depth(subscription_module, pkt_depth):
    """
    Sets early termination packet depth in subscription module.
    :param subscription_module: str path to subscription module to be modified
    :param pkt_depth: int representing maximum packet depth features are collected at, or str 'all'
    :return: None
    """
    old_code = r"fn early_terminate\(&self\) -> bool \{.*?\}"
    if pkt_depth == 'all':
        new_code = r"fn early_terminate(&self) -> bool {\n        false\n    }"
    else:
        new_code = fr"fn early_terminate(&self) -> bool {{\n        self.cnt >= {pkt_depth}\n    }}"
    
    with open(subscription_module, 'r', encoding='utf-8') as file:
        filedata = file.read()

    # find and replace
    if re.search(old_code, filedata, flags=re.DOTALL):
        new_filedata = re.sub(old_code, new_code, filedata, flags=re.DOTALL)
        if filedata != new_filedata:
            print(utils.CYAN + f"{subscription_module}: modified to pkt_depth={pkt_depth}" + utils.RESET)
            with open(subscription_module, 'w', encoding='utf-8') as file:
                file.write(new_filedata)
        else:
            print(utils.CYAN + f"{subscription_module}: no changes" + utils.RESET)
        return True
    return False

def set_freeze_point(subscription_module, pkt_depth):
    """
    Sets freeze point, or the number of packets at which update should no longer continue updating tracked variables. For use in 'app' use case, where in collecting training data, we need to avoid terminating at very small packet depths in order to actually see the TLS SNI, but the features should be "frozen" at the true specified packet depth.
    :param subscription_module: str path to subscription module to be modified
    :param pkt_depth: int representing maximum packet depth features are collected at, or str 'all'
    :return: None
    """
    old_code = r"self\.cnt \+= 1; if .*? { return Ok\(\(\)\) }"
    if pkt_depth == "all":
        new_code = f"self.cnt += 1; if false {{ return Ok(()) }}"
    else:
        new_code = f"self.cnt += 1; if self.cnt > {pkt_depth} {{ return Ok(()) }}"
    with open(subscription_module, 'r', encoding='utf-8') as file:
        filedata = file.read()

    # find and replace
    if re.search(old_code, filedata, flags=re.DOTALL):
        new_filedata = re.sub(old_code, new_code, filedata, flags=re.DOTALL)
        if filedata != new_filedata:
            print(utils.CYAN + f"{subscription_module}: modified to freeze point={pkt_depth}" + utils.RESET)
            with open(subscription_module, 'w', encoding='utf-8') as file:
                file.write(new_filedata)
        else:
            print(utils.CYAN + f"{subscription_module}: no changes" + utils.RESET)
        return True
    return False

def set_filter(main_bin, filter):
    """
    Sets filter.
    :param main_bin: str path to main binary to be modified
    :param filter: RAW str representing raw filter to set. It should appear exactly as it should if you were to manually write it, except with a 'r' character appended before the quotes to signify a raw string.
    :return: None
    """
    old_code = r'#\[filter\("(.*?)"\)\]'
    escaped_filter = filter.replace("\\", "\\\\")
    new_code = f'#[filter("{escaped_filter}")]'
    with open(main_bin, 'r', encoding='utf-8') as file:
        filedata = file.read()

    # find and replace
    if re.search(old_code, filedata, flags=re.DOTALL):
        new_filedata = re.sub(old_code, new_code, filedata, flags=re.DOTALL)
        if filedata != new_filedata:
            print(utils.CYAN + f"{main_bin}: filter modified to {filter}" + utils.RESET)
            with open(main_bin, 'w', encoding='utf-8') as file:
                file.write(new_filedata)
        else:
            print(utils.CYAN + f"{main_bin}: no changes" + utils.RESET)
        return True
    return False


def create_runtime_config(config_template, action, duration=60, buckets=1, log_dir="./log", timing_outfile="./compute_features.csv"):
    """
    Creates a temporary runtime config from a template and sets the output file location for timing measurements.
    :param config_template: str path to template config file
    :param action: str `log`, `extract`, `serve`
    :param duration: int number of seconds to run online
    :param buckets: int number of sink buckets to use
    :param log_dir: str path to where Retina logs should be stored
    :param timing_outfile: str file to output timing measurements
        For `log`, this is just the default './compute_features.csv'
        For `extract`, this is f'{syscost_dir}/compute_features_{feature_decimal}.csv'
        For `serve`, this is just the default './compute_features.csv'
    :return: True if successful, False if failed
    """

    old_text = 'outfile = "./compute_features.csv"'
    new_text = f'outfile = "{timing_outfile}"'
    if action == "log":
        pass
    elif action == "extract":
        if new_text == old_text:
            print(utils.RED + f"Must specify syscost_dir and feature_decimal for `extract`" + utils.RESET)
            return False
    elif action == "serve":
        pass
    else:
        print(utils.RED + f"Must specify 'log', 'extract', 'serve'. Found {action}" + utils.RESET)
        return False
    
    with open(config_template, 'r', encoding='utf-8') as file:
        config_data = toml.load(file)
    
    config_data["timing"]["outfile"] = timing_outfile
    if "online" in config_data:
        config_data["online"]["duration"] = duration
        config_data["online"]["ports"][0]["sink"]["nb_buckets"] = buckets
        config_data["online"]["monitor"]["log"]["directory"] = log_dir
        if action == "log":
            config_data["online"]["ports"][0]["cores"] = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

    print(utils.CYAN + f"Setting config timing output to `{timing_outfile}`" + utils.RESET)
    with open(f"{consts.retina_dir}/scripts/tmp_config.toml", "w", encoding="utf-8") as file:
        toml.dump(config_data, file)
    return True


def compile_binary(feature_comma, action):
    """
    Compiles Retina binary.
    :param feature_comma: str comma-delimited feature set
    :param action: str `log`, `extract`, `serve`
    :return: True if successful, False if failed
    """
    if action == "log":
        compile_features = f"label,collect,{feature_comma}"
        binary = "log_features"
    elif action == "extract":
        compile_features = f"timing,capture_start,{feature_comma}"
        binary = "extract_features"
    elif action == "serve":
        compile_features = f"{feature_comma}"
        binary = "serve_ml"
    else:
        print(utils.RED + f"Must specify 'log', 'extract', 'serve'. Found {action}" + utils.RESET)
        return False
    status = True

    manifest_path = f"{consts.retina_dir}/Cargo.toml"
    cmd = f"cargo build --manifest-path={manifest_path} --release --bin {binary} --features {compile_features}"

    print(utils.CYAN + cmd + utils.RESET)
    popen = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    for stderr_line in iter(popen.stderr.readline, ''):
        if 'Compiling' in stderr_line or 'Finished' in stderr_line:
            print(f'\t> {stderr_line}', end='')
        if 'error' in stderr_line:
            print(utils.RED + f'\t> {stderr_line}', end='' + utils.RESET)
            status = False
    popen.stdout.close()
    popen.stderr.close()
    popen.wait()
    return status

def run_binary(action, outfile, model_file=None):
    """
    Invoke Retina to run feature collection or measurement.
    :param action: str `log`, `extract`, `serve`
    :param outfile: str output file. 
        For `log`, this is the output file for the raw dataset (f'{dataset_dir}/features.jsonl').
        For `extract`, this is the output file containing the config and number of connections processed (f'{syscost_dir}/out_features_{feature_decimal}.json')
        For `serve`, this is the output file containing the config and number of connections processed (f'{syscost_dir}/out_features_{feature_decimal}.json')
    :param model_file: str model file.
        For `log`, this is None
        For `extract`, this is None
        For `serve` this is the path to the file containing the trained model. It MUST be trained with the specified feature set compiled with the binary
    :return: True if successful, False if failed
    """

    if action == "log":
        binary = "log_features"
        executable = f'{consts.retina_dir}/target/release/{binary}'
        config_file = f'{consts.retina_dir}/scripts/tmp_config.toml'
        cmd = f'sudo env LD_LIBRARY_PATH=$LD_LIBRARY_PATH RUST_LOG=info {executable} -c {config_file} -o {outfile}'
    elif action == "extract":
        binary = "extract_features"
        executable = f'{consts.retina_dir}/target/release/{binary}'
        config_file = f'{consts.retina_dir}/scripts/tmp_config.toml'
        cmd = f'sudo env LD_LIBRARY_PATH=$LD_LIBRARY_PATH RUST_LOG=info {executable} -c {config_file} -o {outfile}'
    elif action == "serve":
        binary = "serve_ml"
        executable = f'{consts.retina_dir}/target/release/{binary}'
        config_file = f'{consts.retina_dir}/scripts/tmp_config.toml'
        cmd = f'sudo env LD_LIBRARY_PATH=$LD_LIBRARY_PATH RUST_LOG=error {executable} -c {config_file} -m {model_file} -o {outfile}'
    else:
        print(utils.RED + f"Must specify 'log', 'extract', 'serve'. Found {action}" + utils.RESET)
        return False
    status = True

    print(utils.GREEN + f'> Running `{cmd}`' + utils.RESET)

    EPSILON = 0.0001

    popen = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ''):
        print(stdout_line, end='')
        ## can kill process early if there is loss, or just let it run for measurement
        # if 'SW Dropped' in stdout_line:
        #     num = re.findall('\d*\.*\d+\%', stdout_line)
        #     if not num: continue
        #     value = float(num[0].split('%')[0]) 
        #     if value > EPSILON :
        #         print(utils.RED + f'> TERMINATING, current SW drops {value} greater than {EPSILON}...' + utils.RESET)
        #         stream = os.popen(f'pidof {binary}')
        #         pid = stream.read()
        #         os.system(f'sudo kill -INT {pid}')
        #         status = False     # Failed, skip
    popen.stdout.close()
    popen.wait()
    return status


def collect_raw_dataset(pkt_depth, outfile_name="features.jsonl", duration=600, buckets=64, filter=None, dataset_dir=None):
    """
    Collect raw dataset for all candidate features at specified packet depth.
    :param pkt_depth: int representing maximum packet depth features are collected at, or str 'all'
    :param outfile_name: str name of outfile, defualt 'features.jsonl'
    :param duration: int duration for online, default 600
    :param buckets: int buckets for online, default 64
    :param filter: raw string filter to set, if None don't change anything
    :param dataset_dir: str path to override dataset directory to store collected features
    :return: True if successful, False if failed
    """

    if consts.use_case == "app":
        if pkt_depth != "all" and pkt_depth < 5:
            # for app use case, need to avoid terminating before SNI can be read
            set_freeze_point(f"{consts.retina_dir}/core/src/subscription/features.rs", pkt_depth)
            terminate_depth = 5
        else:
            set_freeze_point(f"{consts.retina_dir}/core/src/subscription/features.rs", "all")
            terminate_depth = pkt_depth
    else:
        terminate_depth = pkt_depth
    
    if filter:
        if not set_filter(f"{consts.retina_dir}/examples/log_features/src/main.rs", filter):
            print(utils.RED + "Failed to set filter" + utils.RESET)
            return False
    
    if not set_pkt_depth(f"{consts.retina_dir}/core/src/subscription/features.rs", terminate_depth):
        print(utils.RED + "Failed to set packet depth" + utils.RESET)
        return False
    config_template = f"{consts.retina_dir}/scripts/base_{consts.use_case}_config.toml"
    if not dataset_dir:
        dataset_dir = os.path.join(consts.dataset_dir, f'pkts_{pkt_depth}')
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)

    log_dir = os.path.join(dataset_dir, "log")
    if not create_runtime_config(config_template, "log", duration=duration, buckets=buckets, log_dir=log_dir, timing_outfile="./compute_features.csv"):
        print(utils.RED + "Failed to create runtime config" + utils.RESET)
        return False
    feature_comma = utils.feature_comma(consts.candidate_features)
    if not compile_binary(feature_comma, "log"):
        print(utils.RED + "Failed to compile binary" + utils.RESET)
        return False

    outfile = os.path.join(dataset_dir, outfile_name)
    if not run_binary("log", outfile):
        print(utils.RED + "Failed to run binary" + utils.RESET)
        return False
    print(utils.GREEN + f"Collected raw dataset at {outfile}" + utils.RESET)
    return True

def measure_costs(feature_set, pkt_depth, duration=60):
    """
    Measure compute costs for given feature set and packet depth.
    :param feature_set: list of feature names
    :param pkt_depth: int representing maximum packet depth features are collected at, or str 'all'
    :param duration: int number of seconds to run
    :return: True if successful, False if failed
    """
    start = time.time()
    if not set_pkt_depth(f"{consts.retina_dir}/core/src/subscription/features.rs", pkt_depth):
        print(utils.RED + "Failed to set packet depth" + utils.RESET)
        return False
    
    config_template = f"{consts.retina_dir}/scripts/base_{consts.use_case}_config.toml"
    syscost_dir = os.path.join(consts.syscost_dir, f'pkts_{pkt_depth}')
    if not os.path.exists(syscost_dir):
        os.mkdir(syscost_dir)
    feature_decimal = utils.feature_decimal(feature_set)
    timing_outfile = os.path.join(syscost_dir, f"compute_features_{feature_decimal}.csv")
    log_dir = os.path.join(syscost_dir, "log")
    if not create_runtime_config(config_template, "extract", duration=duration, buckets=64, log_dir=log_dir, timing_outfile=timing_outfile):
        print(utils.RED + "Failed to create runtime config" + utils.RESET)
        return False
    
    feature_comma = utils.feature_comma(feature_set)
    if not compile_binary(feature_comma, "extract"):
        print(utils.RED + "Failed to compile binary" + utils.RESET)
        return False
    
    outfile = os.path.join(syscost_dir, f"out_features_{feature_decimal}.json")
    if not run_binary("extract", outfile):
        print(utils.RED + "Failed to run binary" + utils.RESET)
        os.system(f'pkill -f top')
        return False
    # terminate may not kill the child top process since it is running with shell=True
    os.system(f'pkill -f top')
    
    print(utils.GREEN + f"Measured {feature_comma} ({feature_decimal}) at {syscost_dir}" + utils.RESET)
    end = time.time()
    print(f"Measure cost duration: {end - start}s")
    return True

def measure_throughput(feature_set, pkt_depth, bucket_range, duration):
    """
    Measure online throughput for given feature set and packet depth.
    :param feature_set: list of feature names
    :param pkt_depth: int representing maximum packet depth features are collected at, or str 'all'
    :param bucket_range: list of bucket values to test
    :param duration: int number of seconds to run
    :return: results directory if successful, None if failed
    """
    if not set_pkt_depth(f"{consts.retina_dir}/core/src/subscription/features.rs", pkt_depth):
        print(utils.RED + "Failed to set packet depth" + utils.RESET)
        return None
    
    config_template = f"{consts.retina_dir}/scripts/base_app_config.toml"
    pkts_throughput_dir = os.path.join(consts.throughput_dir, f'pkts_{pkt_depth}')
    if not os.path.exists(pkts_throughput_dir):
        os.mkdir(pkts_throughput_dir)
    feature_decimal = utils.feature_decimal(feature_set)
    pkts_fts_throughput_dir = os.path.join(pkts_throughput_dir, f"features_{feature_decimal}")
    if not os.path.exists(pkts_fts_throughput_dir):
        os.mkdir(pkts_fts_throughput_dir)

    feature_comma = utils.feature_comma(feature_set)
    if not compile_binary(feature_comma, "serve"):
        print(utils.RED + "Failed to compile binary" + utils.RESET)
        return None
    dt = datetime.datetime.fromtimestamp(time.time())
    ts = dt.strftime('%Y-%m-%d-%H-%M-%S')
    pkts_fts_ts_throughput_dir = os.path.join(pkts_fts_throughput_dir, ts)
    if not os.path.exists(pkts_fts_ts_throughput_dir):
        os.mkdir(pkts_fts_ts_throughput_dir)

    for buckets in bucket_range:
        
        pkts_fts_ts_bkts_throughput_dir = os.path.join(pkts_fts_ts_throughput_dir, f"buckets_{buckets}")
        if not os.path.exists(pkts_fts_ts_bkts_throughput_dir):
            os.mkdir(pkts_fts_ts_bkts_throughput_dir)
        if not create_runtime_config(config_template, "serve", duration=duration, buckets=buckets, log_dir=pkts_fts_ts_bkts_throughput_dir, timing_outfile="./compute_features.csv"):
            print(utils.RED + "Failed to create runtime config" + utils.RESET)
            return None
        
        model_dir = os.path.join(consts.dataset_dir, f"pkts_{pkt_depth}", f"features_{feature_decimal}")
        model_file = os.path.join(model_dir, "rust_dt.bin")
        outfile = os.path.join(pkts_fts_ts_bkts_throughput_dir, f"out_features_{feature_decimal}.json")
        if not run_binary("serve", outfile, model_file):
            print(utils.RED + "Failed to run binary" + utils.RESET)
            return None
        print(utils.GREEN + f"Measured throughput for {feature_comma} ({feature_decimal}) at {pkts_fts_ts_bkts_throughput_dir}" + utils.RESET)
    return pkts_fts_ts_throughput_dir

   
