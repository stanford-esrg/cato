candidate_features = [
     "dur",
    "proto",
    "s_port",
    "d_port",
    "s_load",
    "d_load",
    "s_pkt_cnt",
    "d_pkt_cnt",
    "s_iat_mean",
    "d_iat_mean",
    "tcp_rtt",
    "syn_ack",
    "ack_dat",
    "s_bytes_sum",
    "d_bytes_sum",
    "s_bytes_mean",
    "d_bytes_mean",
    "s_bytes_min",
    "d_bytes_min",
    "s_bytes_max",
    "d_bytes_max",
    "s_bytes_med",
    "d_bytes_med",
    "s_bytes_std",
    "d_bytes_std",
    "s_iat_sum",
    "d_iat_sum",
    "s_iat_min",
    "d_iat_min",
    "s_iat_max",
    "d_iat_max",
    "s_iat_med",
    "d_iat_med",
    "s_iat_std",
    "d_iat_std",
    "s_winsize_sum",
    "d_winsize_sum",
    "s_winsize_mean",
    "d_winsize_mean",
    "s_winsize_min",
    "d_winsize_min",
    "s_winsize_max",
    "d_winsize_max",
    "s_winsize_med",
    "d_winsize_med",
    "s_winsize_std",
    "d_winsize_std",
    "s_ttl_sum",
    "d_ttl_sum",
    "s_ttl_mean",
    "d_ttl_mean",
    "s_ttl_min",
    "d_ttl_min",
    "s_ttl_max",
    "d_ttl_max",
    "s_ttl_med",
    "d_ttl_med",
    "s_ttl_std",
    "d_ttl_std",
    "cwr_cnt",
    "ece_cnt",
    "urg_cnt",
    "ack_cnt",
    "psh_cnt",
    "rst_cnt",
    "syn_cnt",
    "fin_cnt",
]

packet_counters = [
    "s_pkt_cnt", 
    "d_pkt_cnt", 
    "s_bytes_sum", 
    "d_bytes_sum", 
]

packet_times = [
    "s_iat_sum",
    "d_iat_sum",
    "s_iat_min",
    "d_iat_min",
    "s_iat_max",
    "d_iat_max",
    "s_iat_med",
    "d_iat_med",
    "s_iat_std",
    "d_iat_std",
]

tcp_counters = [
    "tcp_rtt",
    "s_winsize_sum",
    "d_winsize_sum",
    "s_winsize_mean",
    "d_winsize_mean",
    "s_winsize_min",
    "d_winsize_min",
    "s_winsize_max",
    "d_winsize_max",
    "s_winsize_med",
    "d_winsize_med",
    "s_winsize_std",
    "d_winsize_std",
    "cwr_cnt",
    "ece_cnt",
    "urg_cnt",
    "ack_cnt",
    "psh_cnt",
    "rst_cnt",
    "syn_cnt",
    "fin_cnt",
]


dataset_dir = "/path/to/dataset/dir"
results_dir = "/path/to/results/dir"
syscost_dir = "/path/to/syscost/dir"
retina_dir = "/path/to/retina"
rust_train_dir = "/path/to/rust_train_dir"
model_type = "dt" # or rf or dnn
use_case = "iot"  # or app or startup


