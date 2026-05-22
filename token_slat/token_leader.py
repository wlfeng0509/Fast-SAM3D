

import math
from contextlib import contextmanager

class TokenLeader:
    def __init__(self):
        self.num_steps = 0                 
        self.resolution = 16              
        self.full_sampling_steps = 6       
        self.full_sampling_end_steps = 19  
        self.assumed_slope = -0.07         
        self.anchor_step = 1                
        self.decoder_resolution = 64       
        self.aggressive_cache_ratio = 0.7   
        self.final_phase_correction_freq = 3   
        
        self.complexity_at_anchor = None   
        self.log_C1 = None                  
        self.b = None                      
        self.schedule_is_set = False       
        self.current_step = 0               


    def set_parameters(self, args):
        self.num_steps = args.effective_steps
        self.resolution = args.resolution
        

        if args.use_token:
            
            self.full_sampling_steps = 2
            self.full_sampling_end_steps = args.full_sampling_end_steps
            self.assumed_slope = args.assumed_slope
            self.anchor_step = args.anchor_step
            self.aggressive_cache_ratio = 0.9
            self.final_phase_correction_freq = 8

        self.current_step = 0
        self.complexity_at_anchor = None
        self.log_C1 = None
        self.b = None
        self.schedule_is_set = False


    def increase_step(self):
        self.current_step += 1


    def record_complexity_at_anchor(self, total_changes):
        if total_changes > 0 and not self.schedule_is_set:
            self.complexity_at_anchor = total_changes
            self.log_C1 = math.log(self.complexity_at_anchor)
            self.b = self.log_C1 - self.assumed_slope * self.anchor_step
            self.schedule_is_set = True


    def get_skip_budget_for_current_step(self, current_t: float) -> int:
        total_tokens = self.total_tokens

        if self.current_step < self.full_sampling_steps:
            return 0

        if self.current_step >= self.full_sampling_end_steps:
            if self.final_phase_correction_freq <= 0: 
                return int(total_tokens * self.aggressive_cache_ratio)

    
            step_in_final_phase = self.current_step - self.full_sampling_end_steps

            if (step_in_final_phase + 1) % self.final_phase_correction_freq == 0:
                return 0  
            else:
                return int(total_tokens * self.aggressive_cache_ratio) 


        if not self.schedule_is_set:
              return 0


        if (self.current_step + 1) % self.final_phase_correction_freq == 0:
                return 0 
        num_to_skip = int(total_tokens * 0.9) 
        return max(0, min(total_tokens, int(num_to_skip)))

