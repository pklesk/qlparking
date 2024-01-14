import numpy as np
from numba import jit

# [SI UNITS ASSUMED EVERYWHERE]

# GENERAL CONSTANTS
CONST_G = 9.80665
CONST_PARKED_MAX_RELATIVE_DISTANCE_DEVIATION = 0.15
CONST_PARKED_MAX_ANGLE_DEVIATION = np.pi / 16.0 # acceptable range of deviation: [-CONST_PARKED_MAX_ANGLE_DEVIATION, CONST_PARKED_MAX_ANGLE_DEVIATION] 

# CAR CONSTANTS
CAR_ACCELERATION_MAGNITUDES_AHEAD = [0.0, 8.0]
CAR_ACCELERATION_MAGNITUDES_BACK = [0.0, 6.0]
CAR_ACCELERATION_MAGNITUDES_SIDE = [0.0, 1.0]
CAR_LENGTH = 4.405
CAR_WIDTH = 1.818
CAR_MU_STATIC = 0.7 
CAR_MU_KINETIC = 0.3  
CAR_MAX_VELOCITY = 150.0 / 3.6  
CAR_MIN_VELOCITY_TO_TURN = 0.75
CAR_MAX_SENSOR_VALUE = 8.0
CAR_N_SENSORS_FRONT = 3
CAR_N_SENSORS_BACK = 3
CAR_N_SENSORS_SIDES = 1
CAR_ANTISTUCK_CHECK_RADIUS = 0.25
CAR_ANTISTUCK_CHECK_SECONDS_BACK = 3.0
CAR_STATE_REPR_FUNCTION_NAME = "dv_flfrblbr"

# PARK PLACE CONSTANTS
PARK_PLACE_LENGTH = 6.10
PARK_PLACE_WIDTH = 2.74

# REWARD CONSTANTS
REWARD_PARKED = 0.0 
REWARD_COLLIDED = -1e2
REWARD_PENALTY_COEF_DISTANCE = 1.0 # can be interpreted as reciprocal of average velocity [m / s] while parking (to estimate time remaining to park)
REWARD_PENALTY_COEF_ANGLE = 0.0 # can be interpreted as estimate of time [s] needed to correct the angle trajectory towards the parking (when wrong by 180 degrees)
REWARD_PENALTY_COEF_GUTTER_DISTANCE = 0.0 # can be interpreted as estimate of time [s] needed to correct one unit of "gutter distance"
# best penalties discovered: (1.0, 32.0, 8.0)

@jit(nopython=True)
def solve_lines_intersection(x11, x12, x21, x22):
    d1x = x12[0] - x11[0]
    d1y = x12[1] - x11[1]
    d2x = x22[0] - x21[0]
    d2y = x22[1] - x21[1]
    denominator = d2x * d1y - d1x * d2y
    if denominator != 0.0:
        t1 = ((x11[0] - x21[0]) * d2y + (x21[1] - x11[1]) * d2x) / denominator
        t2 = ((x11[0] - x21[0]) * d1y + (x21[1] - x11[1]) * d1x) / denominator
    else:         
        t1 = np.inf # parallel lines 
        t2 = np.inf # parallel lines
    return t1, t2

class Car:

    def __init__(self, x=np.array([0.0, 0.0]), angle=0.0,
                 l=CAR_LENGTH, w=CAR_WIDTH, mu_static=CAR_MU_STATIC, mu_kinetic=CAR_MU_KINETIC, 
                 max_velocity=CAR_MAX_VELOCITY, min_velocity_to_turn=CAR_MIN_VELOCITY_TO_TURN, 
                 n_sensors_front=CAR_N_SENSORS_FRONT, n_sensors_back=CAR_N_SENSORS_BACK, n_sensors_sides=CAR_N_SENSORS_SIDES,
                 antistuck_check_radius=CAR_ANTISTUCK_CHECK_RADIUS, antistuck_check_seconds_back=CAR_ANTISTUCK_CHECK_SECONDS_BACK,
                 state_repr_function_name=CAR_STATE_REPR_FUNCTION_NAME):
        self.x_ = x # position        
        self.d_ahead_ = np.array([0.0, 1.0]) # unit direction vector (looking ahead)
        self.d_right_ = np.array([1.0, 0.0]) # unit direction vector (looking right)
        if angle != 0.0:
            rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            self.d_ahead_ = rotation_matrix.dot(self.d_ahead_)
            self.d_right_ = rotation_matrix.dot(self.d_right_)
        self.angle_ahead_ = np.arctan2(self.d_ahead_[1], self.d_ahead_[0])
        if self.angle_ahead_ < 0.0:
            self.angle_ahead_ += 2 * np.pi
        self.l_ = l
        self.w_ = w
        self.mu_static_ = mu_static
        self.mu_kinetic_ = mu_kinetic
        self.max_velocity_ = max_velocity
        self.min_velocity_to_turn_ = min_velocity_to_turn
        self.x_fl_ = None
        self.x_fr_ = None
        self.x_f_ = None
        self.x_bl_ = None
        self.x_br_ = None
        self.x_b_ = None
        self._refresh_corners()
        self.v_ = np.array([0.0, 0.0]) # velocity
        self.v_magnitude_ = 0.0
        self.a_ = np.array([0.0, 0.0]) # acceleration
        self.a_magnitude_ = 0.0
        self.accelerations_imposed_ = []
        self.n_sensors_front_ = n_sensors_front # at least 2
        self.n_sensors_back_ = n_sensors_back # at least 2
        self.n_sensors_sides_ = n_sensors_sides # at least 1
        self.antistuck_check_radius_ = antistuck_check_radius
        self.antistuck_check_seconds_back_ = antistuck_check_seconds_back
        self.state_repr_function_name = state_repr_function_name
        self.state_repr_function = getattr(self, "_state_repr_" + self.state_repr_function_name)        
        self.sensors_front_xs_ = None
        self.sensors_back_xs_ = None
        self.sensors_left_xs_ = None
        self.sensors_right_xs_ = None   
        self._refresh_sensors_xs()
        self.sensors_front_values_ = None
        self.sensors_back_values_ = None
        self.sensors_left_values_ = None
        self.sensors_right_values_ = None
        self.collided_ = False     
        self.collision_x_ = None
        self.to_park_place_f_ = None # vector: car front to target park place front       
        self.to_park_place_b_ = None # vector: car back to target park place back
        self.to_park_place_fl_ = None # vector: car front left to target park place front left        
        self.to_park_place_fr_ = None # vector: car front right to target park place front right       
        self.to_park_place_b_ = None # vector: car back to target park place back
        self.to_park_place_bl_ = None # vector: car back left to target park place back left          
        self.to_park_place_br_ = None # vector: car back right to target park place back right
        self.to_park_place_fl2_ = None # vector: car front to target park place front left                          
        self.to_park_place_fr2_ = None # vector: car front to target park place front right
        self.to_park_place_bl2_ = None # vector: car back to target park place back left           
        self.to_park_place_br2_ = None # vector: car back to target park place back right
        self.distance_ = None # distance between car position (central) and park place position (central)
        self.angle_distance_ = None # angle between car ahead vector and park place ahead vector: [0, pi]       
        self.gutter_distance_ = None
        self.parked_ = False
        self.time_exceeded_ = False
        self.reward_ = None
        self.x_history_ = [] 
        self.x_fl_history_ = []
        self.x_fr_history_ = []
        self.x_bl_history_ = []         
        self.x_br_history_ = []
        self.v_wrd_ = None # vector: car front to target park place front left        
        self.to_park_place_fl2_wrd_ = None # vector: car front to target park place front left                          
        self.to_park_place_fr2_wrd_ = None # vector: car front to target park place front right
        self.to_park_place_bl2_wrd_ = None # vector: car back to target park place back left           
        self.to_park_place_br2_wrd_ = None # vector: car back to target park place back right        
        
    def _refresh_corners(self):
        self.x_fl_ = self.x_ + self.d_ahead_ * 0.5 * self.l_ -  self.d_right_ * 0.5 * self.w_
        self.x_fr_ = self.x_fl_ + self.d_right_ * self.w_
        self.x_f_ = 0.5 * (self.x_fl_ + self.x_fr_)
        self.x_bl_ = self.x_fl_ - self.d_ahead_ * self.l_
        self.x_br_ = self.x_bl_ + self.d_right_ * self.w_
        self.x_b_ = 0.5 * (self.x_bl_ + self.x_br_)
        self.x_r_ = 0.5 * (self.x_fr_ + self.x_br_)
        self.x_l_ = 0.5 * (self.x_fl_ + self.x_bl_)        
        
    def _refresh_sensors_xs(self):
        gap = self.w_ / (self.n_sensors_front_ - 1)   
        self.sensors_front_xs_ = [self.x_fl_ + i * gap * self.d_right_ for i in range(self.n_sensors_front_)]
        gap = self.w_ / (self.n_sensors_back_ - 1)   
        self.sensors_back_xs_ = [self.x_bl_ + i * gap * self.d_right_ for i in range(self.n_sensors_back_)]
        gap = self.l_ / (self.n_sensors_sides_ + 1)   
        self.sensors_left_xs_ = [self.x_bl_ + (i + 1) * gap * self.d_ahead_ for i in range(self.n_sensors_sides_)]
        self.sensors_right_xs_ = [self.x_br_ + (i + 1) * gap * self.d_ahead_ for i in range(self.n_sensors_sides_)]
        
    def _refresh_sensors_values(self, obstacles):
        self.sensors_front_values_ = np.ones(self.n_sensors_front_) * CAR_MAX_SENSOR_VALUE
        self.sensors_back_values_ = np.ones(self.n_sensors_back_) * CAR_MAX_SENSOR_VALUE
        self.sensors_left_values_ = np.ones(self.n_sensors_sides_) * CAR_MAX_SENSOR_VALUE
        self.sensors_right_values_ = np.ones(self.n_sensors_sides_) * CAR_MAX_SENSOR_VALUE
        sensors_info = [(self.sensors_front_xs_, self.sensors_front_values_), 
                        (self.sensors_back_xs_, self.sensors_back_values_),
                        (self.sensors_left_xs_, self.sensors_left_values_),
                        (self.sensors_right_xs_, self.sensors_right_values_)]
        for sensor_xs, sensor_values in sensors_info: 
            for si, sx in enumerate(sensor_xs): 
                for obstacle in obstacles:
                    for oxi in range(len(obstacle.xs_)):
                        ox1 = obstacle.xs_[oxi]
                        ox2 = obstacle.xs_[(oxi + 1) % len(obstacle.xs_)]
                        ts, to = solve_lines_intersection(self.x_, sx, ox1, ox2)
                        if ts >= 0.0 and to >= 0.0 and to <= 1.0:
                            value = np.linalg.norm(ox1 + to * (ox2 - ox1) - sx)
                            if value < sensor_values[si]:
                                sensor_values[si] = value
                                
    def _refresh_to_park_place_vectors(self, park_place):              
        ppf = park_place.x_ + park_place.d_ahead_ * 0.5 * self.l_
        ppb = park_place.x_ - park_place.d_ahead_ * 0.5 * self.l_                
        self.to_park_place_f_ = ppf - self.x_f_
        self.to_park_place_b_ = ppb - self.x_b_
        self.to_park_place_ = 0.5 * (self.to_park_place_f_ + self.to_park_place_b_)        
        self.to_park_place_f_norm_ = np.linalg.norm(self.to_park_place_f_)
        self.to_park_place_b_norm_ = np.linalg.norm(self.to_park_place_b_)
        self.to_park_place_norm_ = np.linalg.norm(self.to_park_place_)
        ppfr = park_place.x_ + park_place.d_ahead_ * 0.5 * self.l_ + park_place.d_right_ * 0.5 * self.w_
        ppfl = ppfr - park_place.d_right_ * self.w_
        ppbr = park_place.x_ - park_place.d_ahead_ * 0.5 * self.l_ + park_place.d_right_ * 0.5 * self.w_
        ppbl = ppbr - park_place.d_right_ * self.w_        
        self.to_park_place_fr_ = ppfr - self.x_fr_
        self.to_park_place_fl_ = ppfl - self.x_fl_
        self.to_park_place_br_ = ppbr - self.x_br_
        self.to_park_place_bl_ = ppbl - self.x_bl_
        self.to_park_place_fr2_ = ppfr - self.x_f_
        self.to_park_place_fl2_ = ppfl - self.x_f_
        self.to_park_place_br2_ = ppbr - self.x_b_
        self.to_park_place_bl2_ = ppbl - self.x_b_
        self.to_park_place_fr2_norm_ = np.linalg.norm(self.to_park_place_fr2_)
        self.to_park_place_fl2_norm_ = np.linalg.norm(self.to_park_place_fl2_)
        self.to_park_place_br2_norm_ = np.linalg.norm(self.to_park_place_br2_)
        self.to_park_place_bl2_norm_ = np.linalg.norm(self.to_park_place_bl2_)
        arg_arccos = max(min(self.d_ahead_.dot(park_place.d_ahead_), 1.0), -1.0)
        self.angle_distance_ = np.arccos(arg_arccos)
        self.distance_ = np.linalg.norm(self.x_ - park_place.x_)
        self.gutter_distance_ = np.abs(park_place.d_right_0_ + park_place.d_right_.dot(self.x_))
        if self.v_magnitude_ == 0.0:
            if self.distance_ <= CONST_PARKED_MAX_RELATIVE_DISTANCE_DEVIATION * park_place.width_ and self.angle_distance_ <= CONST_PARKED_MAX_ANGLE_DEVIATION:
                self.parked_ = True
        if False: # currently pieces of information below inactive (not to slow down computations; for further research on agent-centered state representations)        
            angle = np.arctan2(self.v_[1], self.v_[0]) - self.angle_ahead_
            rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            self.v_wrd_ = rotation_matrix.dot(np.array([0.0, 1.0])) * np.linalg.norm(self.v_)
            angle = np.arctan2(self.to_park_place_fr2_[1], self.to_park_place_fr2_[0]) - self.angle_ahead_
            rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])        
            self.to_park_place_fr2_wrd_ = rotation_matrix.dot(np.array([0.0, 1.0])) * self.to_park_place_fr2_norm_        
            angle = np.arctan2(self.to_park_place_fl2_[1], self.to_park_place_fl2_[0]) - self.angle_ahead_
            rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])                
            self.to_park_place_fl2_wrd_ = rotation_matrix.dot(np.array([0.0, 1.0])) * self.to_park_place_fl2_norm_
            angle = np.arctan2(self.to_park_place_br2_[1], self.to_park_place_br2_[0]) - self.angle_ahead_
            rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])                                                
            self.to_park_place_br2_wrd_ = rotation_matrix.dot(np.array([0.0, 1.0])) * self.to_park_place_br2_norm_
            angle = np.arctan2(self.to_park_place_bl2_[1], self.to_park_place_bl2_[0]) - self.angle_ahead_
            rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            self.to_park_place_bl2_wrd_ = rotation_matrix.dot(np.array([0.0, 1.0])) * self.to_park_place_bl2_norm_
                
    def _check_collisions(self, obstacles):
        car_segments = [(self.x_fl_, self.x_fr_), (self.x_bl_, self.x_br_), (self.x_bl_, self.x_fl_), (self.x_br_, self.x_fr_)]
        for cs1, cs2 in car_segments:  
            for obstacle in obstacles:
                for oxi in range(len(obstacle.xs_)):
                    ox1 = obstacle.xs_[oxi]
                    ox2 = obstacle.xs_[(oxi + 1) % len(obstacle.xs_)]
                    tcs, to = solve_lines_intersection(cs1, cs2, ox1, ox2)
                    if tcs >= 0.0 and tcs <= 1.0 and to >= 0.0 and to <= 1.0:
                        self.collided_ = True
                        self.collision_x_= ox1 + to * (ox2 - ox1)
                        return                
    
    def _refresh_reward(self, dt_since_action, time_remaining=0.0):                                            
        if self.collided_:
            self.reward_ = REWARD_COLLIDED * time_remaining / dt_since_action # 'reward' for collided state is assumed to last until end of episode
        elif self.parked_:
            self.reward_ = REWARD_PARKED * time_remaining / dt_since_action # reward for parked is assumed to last until end of episode
        else:
            self.reward_ = -dt_since_action            
            self.reward_ += -REWARD_PENALTY_COEF_DISTANCE * self.distance_
            self.reward_ += -REWARD_PENALTY_COEF_ANGLE * self.angle_distance_ / np.pi      
            self.reward_ += -REWARD_PENALTY_COEF_GUTTER_DISTANCE * self.gutter_distance_                

    def _state_repr_avms_fb(self):
        v_magnitude_signed = np.sign(self.d_ahead_.dot(self.v_)) * self.v_magnitude_
        return np.concatenate((np.array([self.angle_ahead_, v_magnitude_signed]), self.to_park_place_f_, self.to_park_place_b_))

    def _state_repr_dv_fb(self):
        return np.concatenate((self.d_ahead_, self.v_, self.to_park_place_f_, self.to_park_place_b_))

    def _state_repr_dv_flfrblbr(self):
        return np.concatenate((self.d_ahead_, self.v_, self.to_park_place_fl_, self.to_park_place_fr_, self.to_park_place_bl_, self.to_park_place_br_))

    def _state_repr_dv_flfrblbr2s(self):        
        return np.concatenate((self.d_ahead_, self.v_, self.to_park_place_fl2_, self.to_park_place_fr2_, self.to_park_place_bl2_, self.to_park_place_br2_))
    
    def _state_repr_dv_flfrblbr2s_d(self):        
        return np.concatenate((self.d_ahead_, self.v_, self.to_park_place_fl2_, self.to_park_place_fr2_, self.to_park_place_bl2_, self.to_park_place_br2_, np.array([self.distance_])))

    def _state_repr_dv_flfrblbr2s_da(self):        
        return np.concatenate((self.d_ahead_, self.v_, self.to_park_place_fl2_, self.to_park_place_fr2_, self.to_park_place_bl2_, self.to_park_place_br2_, np.array([self.distance_, self.angle_distance_])))

    def _state_repr_dv_flfrblbr2s_dag(self):        
        return np.concatenate((self.d_ahead_, self.v_, self.to_park_place_fl2_, self.to_park_place_fr2_, self.to_park_place_bl2_, self.to_park_place_br2_, np.array([self.distance_, self.angle_distance_, self.gutter_distance_])))

    def _state_repr_v1_flfrblbr2s_da_wrd(self):        
        return np.concatenate((np.array([self.v_[1]]), self.to_park_place_fl2_wrd_, self.to_park_place_fr2_wrd_, self.to_park_place_bl2_wrd_, self.to_park_place_br2_wrd_, np.array([self.distance_, self.angle_distance_])))
       
    def get_state(self):
        return self.state_repr_function()
                        
    def accelerate_ahead(self, magnitude):
        if not self.collided_ and not self.parked_:
            self.accelerations_imposed_.append(np.copy(self.d_ahead_) * magnitude)
            self.a_ = np.sum(np.array(self.accelerations_imposed_), axis=0)
            self.a_magnitude_ = np.linalg.norm(self.a_)
        
    def accelerate_back(self, magnitude):
        if not self.collided_ and not self.parked_:
            self.accelerations_imposed_.append(-np.copy(self.d_ahead_) * magnitude)        
            self.a_ = np.sum(np.array(self.accelerations_imposed_), axis=0)
            self.a_magnitude_ = np.linalg.norm(self.a_)

    def accelerate_right(self, magnitude):
        if not self.collided_ and not self.parked_:
            if self.v_magnitude_ >= self.min_velocity_to_turn_: 
                self.accelerations_imposed_.append(np.copy(self.d_right_) * magnitude)
                self.a_ = np.sum(np.array(self.accelerations_imposed_), axis=0)
                self.a_magnitude_ = np.linalg.norm(self.a_)
        
    def accelerate_left(self, magnitude):
        if not self.collided_ and not self.parked_:
            if self.v_magnitude_ >= self.min_velocity_to_turn_: 
                self.accelerations_imposed_.append(-np.copy(self.d_right_) * magnitude)
                self.a_ = np.sum(np.array(self.accelerations_imposed_), axis=0)
                self.a_magnitude_ = np.linalg.norm(self.a_)
        
    def step(self, dt, dt_since_action, time_remaining, obstacles, park_place):                     
        # static friction
        mu_static_g = self.mu_static_ * CONST_G
        if self.v_magnitude_ == 0.0 and self.a_magnitude_ > mu_static_g:
            friction_factor_static = mu_static_g / self.a_magnitude_
            self.a_ -= friction_factor_static * self.a_   
            self.a_magnitude_ *= 1.0 - friction_factor_static            
        # kinetic friction
        friction_factor_kinetic = 0.0            
        if self.v_magnitude_ > 0.0:            
            mu_kinetic_g_dt = self.mu_kinetic_ * CONST_G * dt
            v_mean_magnitude = np.linalg.norm(self.v_ + 0.5 * self.a_ * dt)            
            friction_factor_kinetic = min(mu_kinetic_g_dt / v_mean_magnitude, 1.0)                                                
        # update position
        self.x_ += (1.0 - friction_factor_kinetic) * (self.v_ * dt + 0.5 * self.a_ * dt**2)
        # update velocity
        self.v_ = (1.0 - friction_factor_kinetic) * (self.v_ + self.a_ * dt)
        self.v_magnitude_ = np.linalg.norm(self.v_)        
        if self.v_magnitude_ > self.max_velocity_:
            self.v_ = self.max_velocity_ * self.v_ / self.v_magnitude_
            self.v_magnitude_ = self.max_velocity_                
        # update direction vectors                                 
        if self.v_magnitude_ > 0.0:
            d_ahead_old = self.d_ahead_              
            self.d_ahead_ = self.v_ / self.v_magnitude_
            if self.d_ahead_.dot(d_ahead_old) < 0.0:
                self.d_ahead_ *= -1.0 # prevents unrealistic front-back 'flips'
            rotation_matrix = np.array([[0.0, 1.0], [-1.0, 0.0]]) # for -pi/2
            self.d_right_ = rotation_matrix.dot(self.d_ahead_)
            self.angle_ahead_ = np.arctan2(self.d_ahead_[1], self.d_ahead_[0])
            if self.angle_ahead_ < 0.0:
                self.angle_ahead_ += 2 * np.pi                            
        # refresh additional information
        self.accelerations_imposed_ = [] # accelerations for current step are now consumed -> empty list for accelerations fo next step
        self.a_ = np.array([0.0, 0.0])
        self.a_magnitude_ = 0.0
        self._refresh_corners()
        self._refresh_sensors_xs()
        self._refresh_sensors_values(obstacles)
        self._check_collisions(obstacles)
        if self.collided_:
            self.v_ = np.array([0.0, 0.0]) # stop due to collision
            self.v_magnitude_ = 0.0
        if time_remaining - dt <= 0.0:
            self.time_exceeded_ = True         
        self._refresh_to_park_place_vectors(park_place)      
        self._refresh_reward(dt_since_action, time_remaining)
        # memorize some history
        self.x_history_.append(np.copy(self.x_))
        self.x_fl_history_.append(np.copy(self.x_fl_))
        self.x_fr_history_.append(np.copy(self.x_fr_))
        self.x_bl_history_.append(np.copy(self.x_bl_))        
        self.x_br_history_.append(np.copy(self.x_br_))            
        
    def is_stuck(self, dt):
        steps_back = int(self.antistuck_check_seconds_back_ / dt)
        if len(self.x_history_) < steps_back:
            return False        
        return np.linalg.norm(self.x_ - self.x_history_[-steps_back]) <= self.antistuck_check_radius_             
    
class ParkPlace: # not necessarily rectangle (e.g.~parallelogram)
    def __init__(self, x_fl, x_fr, x_bl, x_br):
        self.x_fl_ = x_fl 
        self.x_fr_ = x_fr 
        self.x_bl_ = x_bl 
        self.x_br_ = x_br 
        self.x_ = (self.x_fl_ + self.x_fr_ + self.x_bl_ + self.x_br_) / 4.0
        self.width_ = np.linalg.norm(self.x_fr_ - self.x_fl_)
        self.d_ahead_ = 0.5 * (self.x_fl_ + self.x_fr_) - 0.5 * (self.x_bl_ + self.x_br_)
        self.d_ahead_ /= np.linalg.norm(self.d_ahead_)
        self.d_right_ = 0.5 * (self.x_fr_ + self.x_br_) - 0.5 * (self.x_fl_ + self.x_bl_)
        self.d_right_ /= np.linalg.norm(self.d_right_)
        self.d_right_0_ = -self.d_right_.dot(self.x_)
    
class Obstacle:
    def __init__(self, xs=[]):
        self.xs_ = xs
        
class Scene:
    def __init__(self, dt, car, park_place, obstacles=[]):
        self.car_ = car
        self.park_place_ = park_place
        self.obstacles_ = obstacles
        self.car_._refresh_sensors_values(obstacles)
        self.car_._refresh_to_park_place_vectors(park_place)
        self.car_._refresh_reward(dt)