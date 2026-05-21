from recbole.quick_start import run_recbole

# Correr LightGCN
print("\n=== A correr LightGCN ===")
run_recbole(
    model='LightGCN',
    dataset='emorecsys',
    config_file_list=['lightgcn_config.yaml']
)

# Correr NGCF
print("\n=== A correr NGCF ===")
run_recbole(
    model='NGCF',
    dataset='emorecsys',
    config_file_list=['ngcf_config.yaml']
)