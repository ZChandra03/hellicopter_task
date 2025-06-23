"""Collections of tasks."""

from __future__ import division

import numpy as np


class Trial(object):
    """Class representing a batch of trials."""

    def __init__(self, config, tdim, dataset_size):
        """A batch of trials.
        Args:
            config: dictionary of configurations
            tdim: int, number of time steps
            dataset_size: int, dataset size
        """
        self.float_type = 'float32'
        self.config = config
        self.dt = self.config['dt']
        self.n_input = self.config['n_input']
        self.n_output = self.config['n_output']
        self.dataset_size = dataset_size
        self.tdim = tdim
        self.x = np.zeros((dataset_size, tdim, self.n_input), dtype=self.float_type)
        self.y = np.zeros((dataset_size, tdim, self.n_output), dtype=self.float_type)
        if self.config['loss_type'] == 'mean_squared_error':
            self.y[:, :, :] = 0.05
        self.c_mask = np.zeros((dataset_size, tdim, self.n_output), dtype=self.float_type)
        self.respond = np.zeros(dataset_size, dtype=int)
        self.delay = np.zeros(dataset_size, dtype=int)
        self.std_order = np.zeros(dataset_size, dtype=int)
        self.std_dur = self.config['std_dur']
        self.tone_dur = self.config['tone_dur']
        self.comp_dur = np.zeros(dataset_size, dtype=int)
        self.stim1_ons = np.zeros(dataset_size, dtype=int)
        self.stim1_offs = np.zeros(dataset_size, dtype=int)
        self.stim2_ons = np.zeros(dataset_size, dtype=int)
        self.stim2_offs = np.zeros(dataset_size, dtype=int)
        self._sigma_x = config['sigma_x'] * np.sqrt(2 / config['alpha'])

    def expand(self, var):
        """Expand an int/float to list."""
        if not hasattr(var, '__iter__'):
            var = [var] * self.dataset_size
        return var

    def add(self, loc_type, ons=None, offs=None):
        """Add an input or stimulus output.
        """
        ons = self.expand(ons)
        offs = self.expand(offs)

        for i in range(self.dataset_size):
            if loc_type == 'go_cue':
                self.x[i, offs[i]:offs[i] + 2, 0] = 2
            elif loc_type == 'interval_input':
                self.x[i, ons[i]:ons[i] + 2, 0] = 2
                self.x[i, offs[i]:offs[i] + 2, 0] = 2
            elif loc_type == 'discrim':
                self.y[i, :, 0] = 0
                if self.respond[i]:  # second stim > first stim
                    self.y[i, ons[i]: offs[i], 0] = 0.8
            else:
                raise ValueError('Unknown loc_type')

    def add_x_noise(self):
        """Add input noise."""
        self.x += np.random.RandomState().randn(*self.x.shape) * self._sigma_x

    def add_c_mask(self, pre_offs, post_ons, post_offs):
        """Add a cost mask.

        Usually there are two periods, pre and post response
        Scale the mask weight for the post period so in total it's as important as the pre period
        """
        pre_on = int(100 / self.dt)  # never check the first 100ms
        pre_offs = self.expand(pre_offs)
        post_ons = self.expand(post_ons)
        post_offs = self.expand(post_offs)

        for i in range(self.dataset_size):
            # Post response periods usually have the same length across tasks
            # Pre-response periods usually have different lengths across tasks
            # To keep cost comparable across tasks scale the cost mask of the pre-response period by a factor

            # output 1 = Response/motor/"go" unit
            self.c_mask[i, post_ons[i]:post_offs[i], 0] = 2
            self.c_mask[i, pre_on:pre_offs[i], 0] = 1
            self.c_mask[i, post_offs[i]:, 0] = 2
            self.c_mask[i, pre_offs[i]:post_ons[i], 0] = 0


def interval_discrim(config, mode):
    # Get parameters of task
    dt = config['dt']
    dataset_size = config['dataset_size']
    rng = np.random.RandomState()

    if config['delay']:
        delay = np.full(dataset_size, config['delay'])
    else:
        delay = np.full(dataset_size, 1000)
    if config['std_dur']:
        std_dur = config['std_dur']
    else:
        std_dur = 200  # standard duration interval = onset of tone 1 to onset of tone 2
    if config['tone_dur']:
        tone_dur = config['tone_dur']
    else:
        tone_dur = 2 * dt

    # Experimental modes
    if mode == 'random' or mode == 'remove_first' or mode == 'remove_second' or mode == 'go_cue':
        #comp_dur = rng.choice((50, 100, 150, 250, 300, 350), dataset_size)
        #comp_dur = rng.choice((120, 160, 180, 190, 210, 220, 240, 280), dataset_size)
        #print("called")
        # look for a 'comp_step' in your hp; if present, build symmetric offsets,
        # otherwise fall back to your old hard‑coded list
        step = config.get('comp_step', None)
        if step is not None:
            # generate durations = std_dur + step * [-4, -3, -2, -1, 1, 2, 3, 4]
            offsets = np.array([-4, -3, -2, -1, 1, 2, 3, 4], dtype=int)
            choices = std_dur + offsets * step
        else:
            choices = np.array((120, 160, 180, 190, 210, 220, 240, 280), dtype=int)

        comp_dur = rng.choice(choices, dataset_size)
        std_order = rng.choice([0, 1], dataset_size, p=[0.5, 0.5])  # 0 = std comes first, 1 = std comes second
        int1_offs, int2_ons, int2_offs, respond = np.zeros(dataset_size, dtype=int), np.zeros(dataset_size,
                                                                                              dtype=int), np.zeros(
            dataset_size, dtype=int), np.zeros(dataset_size, dtype=int)
        int1_ons = np.full(dataset_size, 500 / dt).astype(int)
        # int1_ons = (rng.uniform(500, 2000, dataset_size) / dt).astype(int)
        #print("choices", choices)
    elif mode == 'random_onset':
        # comp_dur = rng.choice((50, 100, 150, 250, 300, 350), dataset_size)
        comp_dur = rng.choice((120, 160, 180, 190, 210, 220, 240, 280), dataset_size)
        std_order = rng.choice([0, 1], dataset_size, p=[0.5, 0.5])  # 0 = std comes first, 1 = std comes second
        int1_offs, int2_ons, int2_offs, respond = np.zeros(dataset_size, dtype=int), np.zeros(dataset_size,
                                                                                              dtype=int), np.zeros(
            dataset_size, dtype=int), np.zeros(dataset_size, dtype=int)
        int1_ons = (rng.uniform(500, 2000, dataset_size) / dt).astype(int)
        delay = rng.choice((500, 1000), dataset_size)
    elif mode == 'align stim onset':
        comp_dur = rng.choice((50, 100, 150, 250, 300, 350), dataset_size)
        std_order = rng.choice([0, 1], dataset_size, p=[0, 1])  # 0 = std comes first, 1 = std comes second
        int1_offs, int2_offs, respond = np.zeros(dataset_size, dtype=int), np.zeros(dataset_size, dtype=int), np.zeros(
            dataset_size, dtype=int)
        int1_ons = np.full(dataset_size, 500 / dt).astype(int)
        int2_ons = np.full(dataset_size, 1750 / dt).astype(int)
    elif mode == 'random_delay':
        comp_dur = rng.choice((50, 100, 150, 250, 300, 350), dataset_size)
        #comp_dur = rng.choice((120, 160, 180, 190, 210, 220, 240, 280), dataset_size)
        std_order = rng.choice([0, 1], dataset_size, p=[0.5, 0.5])  # 0 = std comes first, 1 = std comes second
        int1_offs, int2_ons, int2_offs, respond = np.zeros(dataset_size, dtype=int), np.zeros(dataset_size,
                                                                                              dtype=int), np.zeros(
            dataset_size, dtype=int), np.zeros(dataset_size, dtype=int)
        int1_ons = np.full(dataset_size, 500 / dt).astype(int)
        delay = rng.choice((500, 1000), dataset_size)
    else:
        raise ValueError('Unknown mode: ' + str(mode))

    if mode != 'align stim onset':
        for i in range(dataset_size):
            if std_order[i] == 0:
                int1_offs[i] = int(int1_ons[i]) + int(std_dur / dt)
                int2_ons[i] = int(int1_offs[i]) + int(tone_dur / dt) + int(delay[i] / dt)
                int2_offs[i] = int(int2_ons[i]) + int(comp_dur[i] / dt)
                respond[i] = comp_dur[i] > std_dur
            else:
                int1_offs[i] = int(int1_ons[i]) + int(comp_dur[i] / dt)
                int2_ons[i] = int(int1_offs[i]) + int(tone_dur / dt) + int(delay[i] / dt)
                int2_offs[i] = int(int2_ons[i]) + int(std_dur / dt)
                respond[i] = std_dur > comp_dur[i]
    else:
        for i in range(dataset_size):
            if std_order[i] == 0:
                int1_offs[i] = int(int1_ons[i]) + int(std_dur / dt)
                # int2_ons[i] = int(int1_offs[i]) + int(delay / dt)
                int2_offs[i] = int(int2_ons[i]) + int(comp_dur[i] / dt)
                respond[i] = comp_dur[i] > std_dur
            else:
                int1_offs[i] = int(int1_ons[i]) + int(comp_dur[i] / dt)
                # int2_ons[i] = int(int1_offs[i]) + int(delay / dt)
                int2_offs[i] = int(int2_ons[i]) + int(std_dur / dt)
                respond[i] = std_dur > comp_dur[i]

    tdim = int(np.max(int2_offs)) + int(tone_dur / dt) + int(
        2000 / dt)  # max tdim:500+200+1500+350+2000=4550 --> 455 ts

    trial = Trial(config, tdim, dataset_size)
    trial.respond, trial.std_order, trial.delay, trial.std_dur, trial.comp_dur, trial.int1_ons, trial.int1_offs, \
    trial.int2_ons, trial.int2_offs = respond, std_order, delay, std_dur, comp_dur, int1_ons, int1_offs, int2_ons, \
                                      int2_offs

    #check_ons = int2_offs + int(100 / dt)
    check_ons = int2_offs + int(50 / dt)
    if mode != 'remove_first':
        trial.add('interval_input', ons=int1_ons, offs=int1_offs)  # first
    if mode != 'remove_second' and mode != 'go_cue':
        trial.add('interval_input', ons=int2_ons, offs=int2_offs)  # second
    if mode == 'go_cue':
        trial.add('go_cue', ons=int2_ons, offs=int2_offs)

    trial.add('discrim', ons=int2_offs, offs=np.full(dataset_size, tdim))
    trial.add_c_mask(pre_offs=int2_offs, post_ons=check_ons, post_offs=int2_offs + int(1000 / dt))

    return trial


rule_mapping = {'Interval_Discrim': interval_discrim}
rule_name = {'Interval_Discrim': 'Interval_Discrim'}


def generate_trials(rule, hp, mode, noise_on=True):
    """Generate one batch of data.

    Args:
        rule: str, the rule for this batch
        hp: dictionary of hyperparameters
        mode: str, the mode of generating. Options: random, test, psychometric
        noise_on: bool, whether input noise is given

    Return:
        trial: Trial class instance, containing input and target output
    """
    config = hp
    trial = rule_mapping[rule](config, mode)

    if noise_on:
        trial.add_x_noise()

    return trial
