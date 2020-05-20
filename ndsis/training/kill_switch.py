#%%
import os
import datetime
from scannet_config.pathes import kill_switch_path, kill_switch_person
from collections import defaultdict

#%%

class KillSwitch():
    TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
    

    def __init__(
            self, start_epoch, num_epochs, path=kill_switch_path,
            person=kill_switch_person):
        """
        creating this object creates a file at our common kill switch path
        this file is updated during the run and contains the current state of
        the algorithm if the file is deleted or renamed, the check returns
        false and you have to kill the training
        Please trigger the kill switch only use it if the person uses more than
        one gpu!
        """
        super().__init__()
        self.epoch_list = defaultdict(datetime.datetime.now)
        self.num_epochs = num_epochs
        gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '-all')
        self.switch_path = switch_path = os.path.join(
            path, f'{person}_gpu{gpu}.txt')

        with open(self.switch_path,'w') as switch_file:
            self._write_epochs(switch_file, start_epoch)

    def _write_epochs(self, switch_file, epoch):
        self.epoch_list[epoch]
        for epoch, start_time in self.epoch_list.items():
            switch_file.write(
                f'epoch={epoch:>4}/{self.num_epochs}, '
                f'starttime={start_time:{self.TIME_FORMAT}}\n')

    def write_and_check(self, current_epoch, current_step, num_steps):
        """
        updates the file if it still exists
        returns whether the file still exists
        you have to stop the process if it returns false
        """
        if not os.path.exists(self.switch_path):
            return False
        with open(self.switch_path,'w') as switch_file:
            self._write_epochs(switch_file, current_epoch)
            switch_file.write(
                f'current_time={datetime.datetime.now():{self.TIME_FORMAT}}\n')
            switch_file.write(f'step={current_step}/{num_steps}\n')
        return True

#%%
