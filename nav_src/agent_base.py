import json
import os


def _runtime_row_to_traj(row):
    """Convert one entry from runtime.json / submit JSON into internal traj dict."""
    if not row or not row.get('instr_id'):
        return None
    t = dict(row)
    if 'trajectory' in t and 'path' not in t:
        t['path'] = t['trajectory']
    return t


class BaseAgent(object):
    ''' Base class for an REVERIE agent to generate and save trajectories. '''

    def __init__(self, env):
        self.env = env
        self.results = {}

    def get_results(self, detailed_output=False):
        output = []
        for k, v in self.results.items():
            output.append({'instr_id': k, 'trajectory': v['path']})
            if detailed_output:
                output[-1]['details'] = v['details']
                output[-1]['action_plan'] = v['action_plan']
                output[-1]['llm_output'] = v['llm_output']
                output[-1]['llm_thought'] = v['llm_thought']
                output[-1]['llm_observation'] = v['llm_observation']
        return output

    def rollout(self, **args):
        ''' Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]  '''
        raise NotImplementedError

    def _load_runtime_checkpoint(self):
        """Load completed trajectories from logs/runtime.json for resume."""
        path = os.path.join(self.config.log_dir, 'runtime.json')
        if not os.path.isfile(path):
            return {}
        try:
            with open(path, 'r') as f:
                rows = json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
        if not isinstance(rows, list):
            return {}
        out = {}
        for row in rows:
            t = _runtime_row_to_traj(row)
            if t is not None:
                out[t['instr_id']] = t
        return out

    def _merge_partial_submit_preds(self):
        """If preds/submit_<split>.json exists but is incomplete, merge missing instr_ids."""
        name = getattr(self.env, 'name', None)
        if not name or not hasattr(self.env, 'size'):
            return
        path = os.path.join(self.config.pred_dir, 'submit_%s.json' % name)
        if not os.path.isfile(path):
            return
        try:
            with open(path, 'r') as f:
                preds = json.load(f)
        except (json.JSONDecodeError, OSError):
            return
        full_n = (
            self.env.full_dataset_size()
            if hasattr(self.env, 'full_dataset_size')
            else self.env.size()
        )
        if not isinstance(preds, list) or len(preds) >= full_n:
            return
        for row in preds:
            t = _runtime_row_to_traj(row)
            if t is None or 'path' not in t:
                continue
            iid = t['instr_id']
            if iid not in self.results:
                self.results[iid] = t

    @staticmethod
    def get_agent(name):
        return globals()[name+"Agent"]

    def test(self, iters=None, **kwargs):
        # self.env.reset_epoch(shuffle=(iters is not None))   # If iters is not none, shuffle the env batch
        self.losses = []
        self.results = self._load_runtime_checkpoint()
        self._merge_partial_submit_preds()
        n_resume = len(self.results)
        if n_resume:
            print('Resume: loaded %d completed instruction(s) from checkpoint' % n_resume)
        if n_resume and hasattr(self.env, 'restrict_to_pending'):
            n_before = self.env.size()
            self.env.restrict_to_pending(self.results.keys())
            n_after = self.env.size()
            if n_after < n_before:
                print(
                    'Resume: %d episode(s) remaining (excluded completed); order reshuffled'
                    % n_after
                )
        if hasattr(self.env, 'size') and self.env.size() == 0:
            print('Resume: nothing left to run (all instructions already in checkpoint).')
            return
        # We rely on env showing the entire batch before repeating anything
        self.loss = 0
        if iters is not None:
            # For each time, it will run the first 'iters' iterations. (It was shuffled before)
            for i in range(iters):
                for traj in self.rollout(**kwargs):
                    self.loss = 0
                    self.results[traj['instr_id']] = traj
                    preds_detail = self.get_results(detailed_output=True)
                    json.dump(
                    preds_detail,
                    open(os.path.join(self.config.log_dir, 'runtime.json'), 'w'),
                    sort_keys=True, indent=4, separators=(',', ': ')
                    )
        else:   # Do a full round (detect epoch wrap by repeated instr_id order, not merely in results)
            seen_instr_sequence = []
            while True:
                looped = False
                for traj in self.rollout(**kwargs):
                    tid = traj['instr_id']
                    if tid in seen_instr_sequence:
                        looped = True
                    seen_instr_sequence.append(tid)
                    if tid not in self.results:
                        self.loss = 0
                        self.results[tid] = traj
                        preds_detail = self.get_results(detailed_output=True)
                        json.dump(
                        preds_detail,
                        open(os.path.join(self.config.log_dir, 'runtime.json'), 'w'),
                        sort_keys=True, indent=4, separators=(',', ': ')
                        )
                if looped:
                    break
