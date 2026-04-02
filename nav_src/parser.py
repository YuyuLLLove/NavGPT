import argparse
import os

# ---------------------------------------------------------------------------
# OpenAI 兼容 HTTP 接口（百炼 / OpenAI / OpenRouter / 其它兼容网关）
# 换平台时只改下面三项即可；也可用命令行 --api_key / --api_base 覆盖。
# 留空则走环境变量 OPENAI_API_KEY、OPENAI_API_BASE（可选）。
# ---------------------------------------------------------------------------
OPENAI_COMPAT_API_KEY = ""
OPENAI_COMPAT_API_BASE = ""
# 示例 base：
#   https://api.openai.com/v1
#   https://dashscope.aliyuncs.com/compatible-mode/v1
#   https://openrouter.ai/api/v1


def parse_args():
    parser = argparse.ArgumentParser(description="")

    # datasets
    parser.add_argument('--root_dir', type=str, default='../datasets')
    parser.add_argument('--dataset', type=str, default='r2r', choices=['r2r', 'r4r'])
    parser.add_argument('--output_dir', type=str, default='../datasets/R2R/exprs/gpt-3.5-turbo', help='experiment id')
    # parser.add_argument('--output_dir', type=str, default='../datasets/R2R/exprs/LlaMA-2-13b-test', help='experiment id')
    parser.add_argument('--seed', type=int, default=0)

    # Agent
    parser.add_argument('--temperature', type=float, default=0.0, help='temperature for llm')
    parser.add_argument('--llm_model_name', type=str, default='app-cwtopb-1764155124595787091', help='llm model name')
    parser.add_argument(
        '--api_key', type=str, default='',
        help='OpenAI-compatible API key; empty uses OPENAI_COMPAT_API_KEY in parser.py or env',
    )
    parser.add_argument(
        '--api_base', type=str, default='',
        help='OpenAI-compatible base URL (must end with /v1); empty uses code constant or env',
    )
    # parser.add_argument('--llm_model_name', type=str, default='gpt-4', help='llm model name')
    # parser.add_argument('--llm_model_name', type=str, default='LlaMA-2-13b', help='llm model name')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_iterations', type=int, default=10)

    # General config
    parser.add_argument('--iters', type=int, default=10, help='number of iterations to run')
    # parser.add_argument('--iters', type=int, default=None, help='number of iterations to run')
    parser.add_argument('--max_scratchpad_length', type=int, default=1000, help='max number of steps in an episode')
    parser.add_argument('--test', action='store_true', default=False)
    # parser.add_argument('--val_env_name', type=str, default='R2R_val_unseen_instr_0')
    # parser.add_argument('--val_env_name', type=str, default='R2R_val_unseen_instr_1')
    # parser.add_argument('--val_env_name', type=str, default='R2R_val_unseen_instr_2')
    # parser.add_argument('--val_env_name', type=str, default='R2R_val_unseen_instr_3')
    # parser.add_argument('--val_env_name', type=str, default='R2R_val_unseen_instr_4')
    parser.add_argument('--val_env_name', type=str, default='R2R_val_unseen_instr')

    parser.add_argument('--load_instruction', action='store_true', default=True)
    parser.add_argument('--load_action_plan', action='store_true', default=True)

    parser.add_argument('--use_relative_angle', action='store_true', default=True)
    parser.add_argument('--use_history_chain', action='store_true', default=False)
    parser.add_argument('--use_tool_chain', action='store_true', default=False)
    parser.add_argument('--use_navigable', action='store_true', default=False)
    parser.add_argument('--use_single_action', action='store_true', default=True)

    parser.add_argument('--detailed_output', action='store_true', default=True)

    # parser.add_argument('--valid_file', type=str, default='../datasets/R2R/exprs/4-R2R_val_unseen_instr/4-R2R_val_unseen_instr.json', help='valid file name')
    parser.add_argument('--valid_file', type=str, default=None, help='valid file name')

    args, _ = parser.parse_known_args()

    args = postprocess_args(args)

    return args


def postprocess_args(args):
    ROOTDIR = args.root_dir

    # Setup input paths
    args.obs_dir = os.path.join(ROOTDIR, 'R2R', 'observations_list_summarized')
    args.obs_summary_dir = os.path.join(ROOTDIR, 'R2R', 'observations_summarized')
    args.obj_dir = os.path.join(ROOTDIR, 'R2R', 'objects_list')

    args.connectivity_dir = os.path.join(ROOTDIR, 'R2R', 'connectivity')
    args.scan_data_dir = os.path.join(ROOTDIR, 'Matterport3D', 'v1_unzip_scans')

    args.anno_dir = os.path.join(ROOTDIR, 'R2R', 'annotations')
    args.navigable_dir = os.path.join(ROOTDIR, 'R2R', 'navigable')

    # Build paths
    args.log_dir = os.path.join(args.output_dir, 'logs')
    args.pred_dir = os.path.join(args.output_dir, 'preds')

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.pred_dir, exist_ok=True)

    def _nv(s):
        s = (s or '').strip()
        return s or None

    key = _nv(args.api_key) or _nv(OPENAI_COMPAT_API_KEY)
    if not key:
        key = _nv(os.environ.get('OPENAI_API_KEY')) or _nv(
            os.environ.get('DASHSCOPE_API_KEY')
        )
    base = _nv(args.api_base) or _nv(OPENAI_COMPAT_API_BASE)
    if not base:
        base = _nv(os.environ.get('OPENAI_API_BASE'))
    args.api_key = key
    args.api_base = base

    return args

