import os
from typing import Dict

import numpy as np

from evorob.utils.filesys import search_file_list


class EA:
    directory_name = ""
    n_gen = 0
    current_gen = 0
    full_x = []
    full_fitness = []
    x_best_so_far = None
    f_best_so_far = -np.inf
    x = None
    f = None

    def ask(self):
        raise NotImplementedError


    def tell(self, solutions, function_values, save_checkpoint=True):
        raise NotImplementedError

    def save_checkpoint(self):
        curr_gen_path = os.path.join(self.directory_name, str(self.current_gen))
        os.makedirs(curr_gen_path, exist_ok=True)
        np.save(os.path.join(self.directory_name, 'full_f'), np.array(self.full_fitness))
        np.save(os.path.join(self.directory_name, 'full_x'), np.array(self.full_x))
        np.save(os.path.join(curr_gen_path, 'f_best'), np.array(self.f_best_so_far))
        np.save(os.path.join(curr_gen_path, 'x_best'), np.array(self.x_best_so_far))
        np.save(os.path.join(curr_gen_path, 'x'), np.array(self.x))
        np.save(os.path.join(curr_gen_path, 'f'), np.array(self.f))

    def load_checkpoint(self):
        dir_path = search_file_list(self.directory_name, 'f_best.npy')
        assert len(dir_path) > 0, "No files are here, check the directory_name!!"

        self.current_gen = int(dir_path[-1].split('/')[-2])
        curr_gen_path = os.path.join(self.directory_name, str(self.current_gen))
        print(f"Loading from: {curr_gen_path}")
        self.full_fitness = np.load(os.path.join(self.directory_name, 'full_f.npy'))
        self.full_x = np.load(os.path.join(self.directory_name, 'full_x.npy'))
        self.f_best_so_far = np.load(os.path.join(curr_gen_path, 'f_best.npy'))
        self.x_best_so_far = np.load(os.path.join(curr_gen_path, 'x_best.npy'))
        self.x = np.load(os.path.join(curr_gen_path, 'x.npy'))
        self.f = np.load(os.path.join(curr_gen_path, 'f.npy'))
