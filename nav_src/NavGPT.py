import os
import json
import time
from argparse import Namespace

from data_utils import construct_instrs
from utils.logger import write_to_record_file

from utils.data import ImageObservationsDB
from parser import parse_args
from env import R2RNavBatch
from agent import NavAgent

def build_dataset(args):

    feat_db = ImageObservationsDB(args.obs_dir, args.obs_summary_dir, args.obj_dir)

    dataset_class = R2RNavBatch

    val_env_names = [args.val_env_name]

    val_envs = {}
    for split in val_env_names:
        val_instr_data = construct_instrs(
            args.anno_dir, args.dataset, [split]
        )
        val_env = dataset_class(
            feat_db, val_instr_data, args.connectivity_dir, args.navigable_dir,
            batch_size=args.batch_size, seed=args.seed, name=split,
        )   # evaluation using all objects
        val_envs[split] = val_env

    return val_envs


def valid(args, val_envs):

    agent = NavAgent(next(iter(val_envs.values())), args)

    with open(os.path.join(args.log_dir, 'validation_args.json'), 'w') as outf:
        _args = vars(args).copy()
        if _args.get('api_key'):
            _args['api_key'] = '***'
        json.dump(_args, outf, indent=4)
    record_file = os.path.join(args.log_dir, 'valid.txt')
    _log = vars(args).copy()
    if _log.get('api_key'):
        _log['api_key'] = '***'
    write_to_record_file(str(Namespace(**_log)) + '\n\n', record_file)

    for env_name, env in val_envs.items():
        prefix = 'submit'
        submit_path = os.path.join(args.pred_dir, "%s_%s.json" % (prefix, env_name))
        if os.path.exists(submit_path):
            try:
                with open(submit_path) as sf:
                    submit_preds = json.load(sf)
                full_n = (
                    env.full_dataset_size()
                    if hasattr(env, 'full_dataset_size')
                    else env.size()
                )
                if isinstance(submit_preds, list) and len(submit_preds) >= full_n:
                    continue
            except (json.JSONDecodeError, OSError):
                pass
        agent.env = env

        start_time = time.time()
        agent.test(iters=args.iters)
        print(env_name, 'cost time: %.2fs' % (time.time() - start_time))
        # Get the results
        preds = agent.get_results(detailed_output=False)
        # Record llm output details
        if args.detailed_output:
            preds_detail = agent.get_results(detailed_output=True)

            json.dump(
            preds_detail,
            open(os.path.join(args.log_dir, "detail_%s.json" % (env_name)), 'w'),
            sort_keys=True, indent=4, separators=(',', ': ')
            )

        if 'test' not in env_name:
            score_summary, _ = env.eval_metrics(preds)
            loss_str = "Env name: %s" % env_name
            for metric, val in score_summary.items():
                loss_str += ', %s: %.2f' % (metric, val)
            write_to_record_file(loss_str+'\n', record_file)

        json.dump(
            preds,
            open(os.path.join(args.pred_dir, "%s_%s.json" % (prefix, env_name)), 'w'),
            sort_keys=True, indent=4, separators=(',', ': ')
        )
            

def valid_from_file(args, val_envs):

    agent = NavAgent(next(iter(val_envs.values())), args)
    with open(args.valid_file, 'r') as f:
        preds = json.load(f)

    for env_name, env in val_envs.items():
        agent.env = env
        valid_list = [preds]
        for valid_pred in valid_list:
            score_summary, _ = env.eval_metrics(valid_pred)
            loss_str = "Env name: %s, length %d" % (env_name, len(valid_pred))
            for metric, val in score_summary.items():
                loss_str += ', %s: %.2f' % (metric, val)
            print(loss_str)

def main():
    args = parse_args()

    val_envs = build_dataset(args)

    if args.valid_file is not None:
        valid_from_file(args, val_envs)
    else:
        valid(args, val_envs)
            

if __name__ == '__main__':
    main()
