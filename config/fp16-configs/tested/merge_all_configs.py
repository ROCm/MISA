import os
import argparse

if __name__ == '__main__':
    argsParser = argparse.ArgumentParser()
    argsParser.add_argument("config_file_root", help='give me a config files root')
    args = argsParser.parse_args()
    config_file_root = args.config_file_root
    merged_config_file = os.path.join(config_file_root, 'igemm_fwd_gtc_gfx908_fp16.config')

    if os.path.exists(merged_config_file):
        os.remove(merged_config_file)
    for _, _, config_files in os.walk(config_file_root):
        with open(merged_config_file, 'w') as f_out:
            for config_file in config_files:
                if config_file.endswith('.config'):
                    each_config_file = os.path.join(config_file_root, config_file)
                    with open(each_config_file, 'r') as f_in:
                        config_txt = f_in.read()
                        f_out.write(config_txt)
