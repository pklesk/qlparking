import defs
from defs import Car, Obstacle, ParkPlace, Scene 
import numpy as np
import pygame
import sys
import time
import itertools
import pickle
from copy import deepcopy
from qapproximations import QRidgeRegressor, QMLPRegressor, QMLPRegressorShared
from sklearn.preprocessing import PolynomialFeatures
import ctypes
import zipfile as zf
import os

# ACTIONS-, PARK PLACE-RELATED CONSTANTS
ACCELERATION_MAGNITUDES_AHEAD = defs.CAR_ACCELERATION_MAGNITUDES_AHEAD
ACCELERATION_MAGNITUDES_BACK = defs.CAR_ACCELERATION_MAGNITUDES_BACK
ACCELERATION_MAGNITUDES_SIDE = defs.CAR_ACCELERATION_MAGNITUDES_SIDE
ACTION_PAIRS = list(itertools.product([-1, 0, 1], repeat=2))
ACTION_PAIRS_INDEXER = dict(zip(ACTION_PAIRS, range(len(ACTION_PAIRS))))
PARK_PLACE_LENGTH = defs.PARK_PLACE_LENGTH
PARK_PLACE_WIDTH = defs.PARK_PLACE_WIDTH

# MAIN SETTINGS: LEARNING OR TESTING
LEARNING_ON = False # if False then testing mode
TEST_MODEL_NAME = "0726961302" # string name equal to hash code e.g. "2354513149"(without "_q.bin" suffix)
TEST_SCENE_FUNCTION_NAME = None # if None then equivalent to QL_SCENE_FUNCTION_NAME 
TEST_RANDOM_SEED = 1
TEST_EPI_SEEDS = []# list of test seeds for demo, if not specified then TEST_RANDOM_SEED applied to generate seeds for episodes
TEST_N_EPISODES = 1000 
TEST_ANIMATION_ON = False
TEST_EPS = 0.0
FOLDER_MODELS = "../models/"
FOLDER_MODELS_ZIPPED = "../models_zipped/"
FOLDER_EXTRAS = "../extras/"
EXPERIENCE_BUFFER_MAX_SIZE = int(5 * 10**7)
LEARNING_QUALITY_OBSERVATIONS_EMAS_DECAY = 0.995
DEMO_TITLE_LINE_1 = None 
DEMO_TITLE_LINE_2 = None 

# DICTIONARIES OF PREDEFINED: TRANSFORMERS, APPROXIMATORS
TRANSFORMERS = {
    "poly_1": PolynomialFeatures(degree=1, include_bias=False),
    "poly_2": PolynomialFeatures(degree=2, include_bias=False),
    "poly_3": PolynomialFeatures(degree=3, include_bias=False),
    "poly_4": PolynomialFeatures(degree=4, include_bias=False)
    }
APPROXIMATORS = {
    "qmlp_small": QMLPRegressor(n_actions=len(ACTION_PAIRS), hidden_layer_sizes=(256, 128, 64, 32), batch_size=128, learning_rate=1e-4, random_state=0, ema_decay=0.0),
    "qmlp_medium": QMLPRegressor(n_actions=len(ACTION_PAIRS), hidden_layer_sizes=(512, 128, 64, 32), batch_size=128, learning_rate=1e-4, random_state=0, ema_decay=0.0),
    "qmlp_large": QMLPRegressor(n_actions=len(ACTION_PAIRS), hidden_layer_sizes=(512, 256, 128, 64), batch_size=128, learning_rate=1e-4, random_state=0, ema_decay=0.0),
    "qmlp_large_deeper": QMLPRegressor(n_actions=len(ACTION_PAIRS), hidden_layer_sizes=(512, 256, 128, 64, 32), batch_size=128, learning_rate=1e-4, random_state=0, ema_decay=0.0),
    "qmlp_very_large": QMLPRegressor(n_actions=len(ACTION_PAIRS), hidden_layer_sizes=(1024, 512, 256, 128), batch_size=128, learning_rate=1e-4, random_state=0, ema_decay=0.0),
    "qmlp_shared_small": QMLPRegressorShared(n_actions=len(ACTION_PAIRS), hidden_layer_sizes=(256, 128, 64, 32), batch_size=128, learning_rate=1e-3, random_state=0, ema_decay=0.0),
    "qmlp_shared_medium": QMLPRegressorShared(n_actions=len(ACTION_PAIRS), hidden_layer_sizes=(512, 128, 64, 32), batch_size=128, learning_rate=1e-3, random_state=0, ema_decay=0.0),
    "qmlp_shared_large": QMLPRegressorShared(n_actions=len(ACTION_PAIRS), hidden_layer_sizes=(512, 256, 128, 64), batch_size=128, learning_rate=1e-3, random_state=0, ema_decay=0.0),
    "qmlp_shared_large_deeper": QMLPRegressorShared(n_actions=len(ACTION_PAIRS), hidden_layer_sizes=(512, 256, 128, 64, 32), batch_size=128, learning_rate=1e-3, random_state=0, ema_decay=0.0),
    "qmlp_shared_very_large": QMLPRegressorShared(n_actions=len(ACTION_PAIRS), hidden_layer_sizes=(2048, 1024, 512, 256), batch_size=128, learning_rate=1e-3, random_state=0, ema_decay=0.0),    
    "qridge_1e1_090": QRidgeRegressor(n_actions=len(ACTION_PAIRS), l2_penalty=1e1, fit_intercept=True, use_numba=True, ema_decay=0.90),    
    "qridge_1e2_095": QRidgeRegressor(n_actions=len(ACTION_PAIRS), l2_penalty=1e2, fit_intercept=True, use_numba=True, ema_decay=0.95),    
    "qridge_1e3_095": QRidgeRegressor(n_actions=len(ACTION_PAIRS), l2_penalty=1e3, fit_intercept=True, use_numba=True, ema_decay=0.95),
    "qridge_1e4_095": QRidgeRegressor(n_actions=len(ACTION_PAIRS), l2_penalty=1e4, fit_intercept=True, use_numba=True, ema_decay=0.95)
    }

# CONSTANTS FOR Q-LEARNING AND SIMULATION 
QL_RANDOM_SEED = 0
QL_DT = 0.025 # [s]
QL_FPS = int(round(1.0 / QL_DT))
QL_ANIMATION_ON = True
QL_ANIMATION_FREQUENCY = 250
QL_STEERING_GAP_STEPS = 4
QL_N_EPISODES = 1 * 10**4  
QL_EPISODE_TIME_LIMIT = 25.0 # [s]
QL_COLLECT_EXPERIENCE_PROBABILITY = 1.0
QL_GAMMA = 0.99
QL_EPS_MAX = 0.5
QL_EPS_MIN = 0.1
QL_EPS_MIN_AT_EPISODE = QL_N_EPISODES
QL_FIT_GAP_EPISODES = 20
QL_FIRST_FIT_AT_EPISODE = 200 
QL_FIT_BATCH_SIZE = 4 * 65536
QL_ORACLE_SWITCHING = True
QL_ORACLE_FIRST_SWITCH_AFTER_EPISODE = 1000
QL_ORACLE_SWITCH_GAP_EPISODES = 500  
QL_ORACLE_SLOW_UPDATES_DECAY = 1.0 # 1.0 means no slow updates take place (only hard switching)
QL_ANTISTUCK_NUDGE = True
QL_ANTISTUCK_NUDGE_STEERING_STEPS = 2
QL_SCENE_FUNCTION_NAME = "pp_middle_obstacles_oppdist_1_side_10_angle_pi"   
QL_TRANSFORMER = TRANSFORMERS["poly_1"]
QL_APPROXIMATOR = APPROXIMATORS["qmlp_large"] # "qmlp_small"
QL_INITIAL_MODEL_NAME = None # for incremental learning, without extension

# DRAWING CONSTANTS
SCREEN_RESOLUTION = (720, 720)
SCREEN_RESOLUTION_DPI_AWARE = False
SCENE_X_RANGE = (-17.5, 17.5) # [m]
SCENE_Y_RANGE = (-17.5, 17.5) # [m]
SCALER_X_A = ((SCREEN_RESOLUTION[0] - 1) - 0) / (SCENE_X_RANGE[1] - SCENE_X_RANGE[0])
SCALER_X_B = 0 - SCALER_X_A * SCENE_X_RANGE[0]
SCALER_Y_A = (0 - (SCREEN_RESOLUTION[1] - 1)) / (SCENE_Y_RANGE[1] - SCENE_Y_RANGE[0])
SCALER_Y_B = 0 - SCALER_Y_A * SCENE_Y_RANGE[1]
SCALER_A = np.array([SCALER_X_A, SCALER_Y_A])
SCALER_B = np.array([SCALER_X_B, SCALER_Y_B])  
COLOR_CAR = (176, 196, 222) 
COLOR_CAR_BORDER = (0, 0, 160)
COLOR_V_VECTOR = (0, 120, 0)
COLOR_A_VECTOR = (160, 0, 0)
COLOR_Q_VALUE = (64, 64, 255)
COLOR_OBSTACLE = (0, 0, 0)
COLOR_PARK_PLACE = (65, 105, 225)
COLOR_SENSOR_BEAM = (0, 206, 209)
COLOR_TO_PARK_PLACE_VECTOR = (180, 180, 180)
COLOR_TRACE_FRONT = (218, 165, 32)
COLOR_TRACE_BACK = (154, 205, 50)
COLOR_COLLISION = (255, 0, 0)
COLOR_PARKED = (0, 0, 255)
COLOR_TIME_LIMIT_EXCEEDED = (255, 99, 71)
COLOR_TEXT = (0, 0, 0)
COLOR_TEXT_HIGHLIGHT = (255, 0, 0)
COLOR_TEXT_INTRO = (255, 255, 255)
COLOR_INITIAL_AREA_AND_ANGLE = (0, 0, 0)
LINE_WIDTH_CAR = 2
LINE_WIDTH_TRACE = 1
LINE_WIDTH_V_VECTOR = 3
LINE_WIDTH_A_VECTOR = 3
LINE_WIDTH_SENSOR = 2
LINE_WIDTH_SENSOR_BEAM = 1
LINE_WIDTH_OBSTACLE = 4
LINE_WIDTH_PARK_PLACE = 1
LINE_WIDTH_COLLISION = 2
LINE_WIDTH_TO_PARK_PLACE_VECTOR = 1 
RADIUS_SENSOR = 2
RADIUS_COLLISION = 10
DRAWING_FACTOR_A = 2.0
TEXT_FONT_NAME = "lucidaconsole"   
TEXT_FONT_SIZE = 10
TEXT_MARGIN = 4
TEXT_MESSAGE_FONT_SIZE = 36
TEXT_INTRO_FONT_NAME = "bahnschrift"
TEXT_INTRO_FONT_SIZE = 10
TEXT_INTRO_MESSAGE_FONT_SIZE = 36
PLOT_FONTSIZE_SUPTITLE = 13
PLOT_FONTSIZE_TITLE = 9.5
PLOT_FONTSIZE_AXES = 12.5
PLOT_FONTSIZE_LEGEND = 9.5
PLOT_FIGSIZE = (10, 6.5)
PLOT_MARKERSIZE = 4
PLOT_GRID_COLOR = (0.4, 0.4, 0.4) 
PLOT_GRID_DASHES = (4.0, 4.0)
PLOT_LEGEND_LOC = "best"
PLOT_LEGEND_HANDLELENGTH = 4
PLOT_LEGEND_LABELSPACING = 0.1

def hash_function(s):
    h = 0
    for c in s:
        h *= 31 
        h += ord(c)
    return h

def experiment_params():
    params = {}
    for key in dir(defs):
        if key.startswith("CONST_") or key.startswith("CAR_") or key.startswith("PARK_PLACE_") or key.startswith("REWARD_"):
            params["DEFS_" + key] = getattr(defs, key)        
    for key, value in globals().items():
        if key.startswith("QL_"):
            params[key] = value
    keys = list(params.keys())
    keys.sort()
    params_sorted = {key: params[key] for key in keys}
    return params_sorted

def experiment_hash_str(digits=10):
    params_sorted = experiment_params()
    return str((hash_function(str(params_sorted)) & ((1 << 32) - 1)) % 10**digits).rjust(digits, "0") 

def dict_to_str(d):
    dict_str = "{"
    for i, key in enumerate(d):
        dict_str += "\n  "  + str(key) + ": " + str(d[key]) + ("," if i < len(d) - 1 else "")    
    dict_str += "\n}"
    return dict_str

def pickle_all(fname, some_list):
    print(f"PICKLE... [{fname}]")
    t1 = time.time()
    f = open(fname, "wb+")
    pickle.dump(some_list, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
    t2 = time.time()
    print(f"PICKLE DONE. [time: {t2 - t1} s]")

def unpickle_all(fname):
    print(f"UNPICKLE... [{fname}]")
    t1 = time.time()    
    f = open(fname, "rb")
    some_list = pickle.load(f)
    f.close()
    t2 = time.time()
    print(f"UNPICKLE DONE. [time: {t2 - t1} s]")
    return some_list

def zip_models():
    print(f"ZIP MODELS...")
    t1 = time.time()
    dir_list = os.listdir(FOLDER_MODELS)
    models_dict = {}
    for fname in dir_list:
        prefix = fname[:fname.find("_")]
        if prefix not in models_dict:
            models_dict[prefix] = []
        models_dict[prefix].append(fname)    
    for i, key in enumerate(models_dict):                                 
        for fname in models_dict[key]:            
            fpath = FOLDER_MODELS + fname
            target_path = FOLDER_MODELS_ZIPPED + fname + ".zip"
            print(f"[{i + 1}/{len(models_dict)}: zipping file {fpath} to {target_path}]")
            with zf.ZipFile(target_path, mode="w", compression=zf.ZIP_DEFLATED) as archive:
                archive.write(fpath, arcname=fname)            
    t2 = time.time()
    print(f"ZIP MODELS DONE. [time: {t2 - t1} s]")         

def draw_dashed_line(surface, color, start_pos, end_pos, dash_length=4):
    x1, y1 = start_pos
    x2, y2 = end_pos
    dx = x2 - x1
    dy = y2 - y1
    distance = max(abs(dx), abs(dy))
    dx = dx / distance
    dy = dy / distance
    for i in range(int(distance/dash_length)):
        if i % 2 == 0:
            pygame.draw.line(surface, color, (x1 + dx * i * dash_length, y1 + dy * i * dash_length),
                             (x1 + dx * (i + 1) * dash_length, y1 + dy * (i + 1) * dash_length), 1)

def draw_intro(screen, ehs, title_line_1, title_line_2=None, smaller_title_line_2=True):
    screen.fill((0, 0, 0))    
    font = pygame.font.SysFont(TEXT_INTRO_FONT_NAME, TEXT_INTRO_MESSAGE_FONT_SIZE)     
    title_current_y = SCREEN_RESOLUTION[1] // 2 - 1 * TEXT_INTRO_MESSAGE_FONT_SIZE
    text_img = font.render(title_line_1, True, COLOR_TEXT_INTRO)
    text_rect = text_img.get_rect(center=(SCREEN_RESOLUTION[0] // 2, title_current_y))
    screen.blit(text_img, text_rect)
    title_current_y += TEXT_INTRO_MESSAGE_FONT_SIZE
    if title_line_2 is not None:
        if smaller_title_line_2:
            font = pygame.font.SysFont(TEXT_INTRO_FONT_NAME, TEXT_INTRO_MESSAGE_FONT_SIZE // 2)
        text_img = font.render(title_line_2, True, COLOR_TEXT_INTRO)
        text_rect = text_img.get_rect(center=(SCREEN_RESOLUTION[0] // 2, title_current_y))
        screen.blit(text_img, text_rect)
        title_current_y += TEXT_INTRO_MESSAGE_FONT_SIZE
    font = pygame.font.SysFont(TEXT_INTRO_FONT_NAME, TEXT_INTRO_FONT_SIZE)         
    y_offset = int(0.875 * SCREEN_RESOLUTION[1])
    text_img = font.render(f"experiment hash code: {ehs}", True, COLOR_TEXT_INTRO)
    screen.blit(text_img, (TEXT_MARGIN, y_offset))
    y_offset += TEXT_FONT_SIZE
    test_model_name = TEST_MODEL_NAME if TEST_MODEL_NAME is not None else "-"
    text_img = font.render(f"test model name: {test_model_name}", True, COLOR_TEXT_INTRO)
    screen.blit(text_img, (TEXT_MARGIN, y_offset))
    y_offset += TEXT_FONT_SIZE
    text_img = font.render(f"reward parameters: ({defs.REWARD_PENALTY_COEF_DISTANCE}, {defs.REWARD_PENALTY_COEF_ANGLE}, {defs.REWARD_PENALTY_COEF_GUTTER_DISTANCE})", True, COLOR_TEXT_INTRO)
    screen.blit(text_img, (TEXT_MARGIN, y_offset))
    y_offset += TEXT_FONT_SIZE
    text_img = font.render(f"state representation: {defs.CAR_STATE_REPR_FUNCTION_NAME}", True, COLOR_TEXT_INTRO)
    screen.blit(text_img, (TEXT_MARGIN, y_offset))
    y_offset += TEXT_FONT_SIZE    
    text_img = font.render(f"Q models: {QL_APPROXIMATOR}", True, COLOR_TEXT_INTRO)
    screen.blit(text_img, (TEXT_MARGIN, y_offset))
    y_offset += TEXT_FONT_SIZE
    text_img = font.render(f"scene for learning: {QL_SCENE_FUNCTION_NAME} [seed: {QL_RANDOM_SEED}]", True, COLOR_TEXT_INTRO)
    screen.blit(text_img, (TEXT_MARGIN, y_offset))
    y_offset += TEXT_FONT_SIZE
    text_img = font.render(f"scene for testing: {TEST_SCENE_FUNCTION_NAME} [seed: {TEST_RANDOM_SEED}]", True, COLOR_TEXT_INTRO)
    screen.blit(text_img, (TEXT_MARGIN, y_offset))
    y_offset += TEXT_FONT_SIZE
    if TEST_EPI_SEEDS is not None and len(TEST_EPI_SEEDS) > 0:
        text_img = font.render(f"seeds of particular episodes in this demo: {TEST_EPI_SEEDS}", True, COLOR_TEXT_INTRO)
        screen.blit(text_img, (TEXT_MARGIN, y_offset))
        y_offset += TEXT_FONT_SIZE             

def draw_scene(screen, scene, time_elapsed, Q_pred):
    screen.fill((255, 255, 255))
    pygame.draw.rect(screen, (0, 0, 0), pygame.Rect(0, 0, SCREEN_RESOLUTION[0], SCREEN_RESOLUTION[1]),  1)            
    lines_function = pygame.draw.aalines # or: pygame.draw.lines (anti-aliased)                
    # drawing park place
    pps = [tuple(SCALER_A * scene.park_place_.x_fl_ + SCALER_B), tuple(SCALER_A * scene.park_place_.x_fr_ + SCALER_B), tuple(SCALER_A * scene.park_place_.x_br_ + SCALER_B), tuple(SCALER_A * scene.park_place_.x_bl_ + SCALER_B)]
    pygame.draw.lines(screen, COLOR_PARK_PLACE, True, pps, LINE_WIDTH_PARK_PLACE)
    font = pygame.font.SysFont(TEXT_FONT_NAME, TEXT_FONT_SIZE)
    fc = 0.5 * (scene.park_place_.x_fl_ + scene.park_place_.x_fr_)
    bc = 0.5 * (scene.park_place_.x_bl_ + scene.park_place_.x_br_)
    letters_shift = 0.15
    fs = SCALER_A * ((1.0 - letters_shift) * fc + letters_shift * bc) + SCALER_B
    bs = SCALER_A * (letters_shift * fc + (1.0 - letters_shift) * bc) + SCALER_B    
    text_img = font.render("F", True, COLOR_PARK_PLACE)
    text_rect = text_img.get_rect(center=(fs[0], fs[1]))
    screen.blit(text_img, text_rect)
    text_img = font.render("B", True, COLOR_PARK_PLACE)
    text_rect = text_img.get_rect(center=(bs[0], bs[1]))
    screen.blit(text_img, text_rect)                    
    # drawing obstacles
    for obstacle in scene.obstacles_:
        os = [tuple(SCALER_A * x + SCALER_B) for x in obstacle.xs_]
        lines_function(screen, COLOR_OBSTACLE, True, os, LINE_WIDTH_OBSTACLE)        
    # drawing car
    car = scene.car_    
    # car trace (history)
    trace_seconds_back = 25.0
    trace_back = min(int(np.round(trace_seconds_back * 1.0 / QL_DT)), len(car.x_history_))
    for i in range(1, trace_back):
        i_new = -i
        i_old = -(i + 1)
        fls_new = SCALER_A * car.x_fl_history_[i_new] + SCALER_B
        fls_old = SCALER_A * car.x_fl_history_[i_old] + SCALER_B        
        lines_function(screen, COLOR_TRACE_FRONT, True, [tuple(fls_new), tuple(fls_old)], LINE_WIDTH_TRACE)
        frs_new = SCALER_A * car.x_fr_history_[i_new] + SCALER_B
        frs_old = SCALER_A * car.x_fr_history_[i_old] + SCALER_B        
        lines_function(screen, COLOR_TRACE_FRONT, True, [tuple(frs_new), tuple(frs_old)], LINE_WIDTH_TRACE)                
        bls_new = SCALER_A * car.x_bl_history_[i_new] + SCALER_B
        bls_old = SCALER_A * car.x_bl_history_[i_old] + SCALER_B        
        lines_function(screen, COLOR_TRACE_BACK, True, [tuple(bls_new), tuple(bls_old)], LINE_WIDTH_TRACE)
        brs_new = SCALER_A * car.x_br_history_[i_new] + SCALER_B
        brs_old = SCALER_A * car.x_br_history_[i_old] + SCALER_B        
        lines_function(screen, COLOR_TRACE_BACK, True, [tuple(brs_new), tuple(brs_old)], LINE_WIDTH_TRACE)
    # drawing vectors to park place
    if not car.collided_:        
        tpp_info = [(car.x_fl_, car.to_park_place_fl_), (car.x_fr_, car.to_park_place_fr_), (car.x_bl_, car.to_park_place_bl_), (car.x_br_, car.to_park_place_br_)]
        for tpp_corner, tpp_vector in tpp_info:
            tpp_start = SCALER_A * tpp_corner + SCALER_B
            tpp_end = SCALER_A * (tpp_corner + tpp_vector) + SCALER_B        
            lines_function(screen, COLOR_TO_PARK_PLACE_VECTOR, False, [tuple(tpp_start), tuple(tpp_end)], LINE_WIDTH_TO_PARK_PLACE_VECTOR)                
    # car corners coords
    fl = car.x_ + car.d_ahead_ * 0.5 * car.l_ - car.d_right_ * 0.5 * car.w_
    fr = fl + car.d_right_ * car.w_ 
    bl = fl - car.d_ahead_ * car.l_
    br = bl + car.d_right_ * car.w_
    f = 0.5 * (fl + fr)    
    # car corners screen coords
    fls = SCALER_A * fl + SCALER_B
    frs = SCALER_A * fr + SCALER_B
    bls = SCALER_A * bl + SCALER_B
    brs = SCALER_A * br + SCALER_B
    fs = SCALER_A * f + SCALER_B        
    pygame.draw.polygon(screen, COLOR_CAR, [tuple(fls), tuple(frs), tuple(brs), tuple(bls)], 0)
    pygame.draw.polygon(screen, COLOR_CAR, [tuple(bls), tuple(fs), tuple(brs)], 0)
    lines_function(screen, COLOR_CAR_BORDER, True, [tuple(fls), tuple(frs), tuple(brs), tuple(bls)], LINE_WIDTH_CAR)
    lines_function(screen, COLOR_CAR_BORDER, True, [tuple(bls), tuple(fs), tuple(brs)], LINE_WIDTH_CAR)                
    # sensor points
    for x in car.sensors_front_xs_:
        pygame.draw.circle(screen, COLOR_CAR, SCALER_A * x + SCALER_B, RADIUS_SENSOR, LINE_WIDTH_SENSOR)
    for x in car.sensors_back_xs_:
        pygame.draw.circle(screen, COLOR_CAR, SCALER_A * x + SCALER_B, RADIUS_SENSOR, LINE_WIDTH_SENSOR)        
    for x in car.sensors_left_xs_:
        pygame.draw.circle(screen, COLOR_CAR, SCALER_A * x + SCALER_B, RADIUS_SENSOR, LINE_WIDTH_SENSOR)
    for x in car.sensors_right_xs_:
        pygame.draw.circle(screen, COLOR_CAR, SCALER_A * x + SCALER_B, RADIUS_SENSOR, LINE_WIDTH_SENSOR)    
    # drawing a, v vectors of car
    xs = np.round(SCALER_A * car.x_ + SCALER_B).astype(np.int32)
    aas = np.round(SCALER_A * (car.x_ + DRAWING_FACTOR_A * car.a_) + SCALER_B).astype(np.int32)
    vs = np.round(SCALER_A * (car.x_ + car.v_) + SCALER_B).astype(np.int32)    
    pygame.draw.lines(screen, COLOR_A_VECTOR, False, [tuple(xs), tuple(aas)], LINE_WIDTH_A_VECTOR)
    pygame.draw.lines(screen, COLOR_V_VECTOR, False, [tuple(xs), tuple(vs)], LINE_WIDTH_V_VECTOR)
    if not car.collided_:
        # drawing sensor beams
        sensors_info = [(car.sensors_front_xs_, car.sensors_front_values_), 
                        (car.sensors_back_xs_, car.sensors_back_values_),
                        (car.sensors_left_xs_, car.sensors_left_values_),
                        (car.sensors_right_xs_, car.sensors_right_values_)]
        for sensor_xs, sensor_values in sensors_info:    
            for si in range(len(sensor_xs)):
                beam_vector = sensor_xs[si] - car.x_
                beam_vector *= sensor_values[si] / np.linalg.norm(beam_vector)
                beam_start = SCALER_A * sensor_xs[si] + SCALER_B
                beam_end = SCALER_A * (sensor_xs[si] + beam_vector) + SCALER_B
                lines_function(screen, COLOR_SENSOR_BEAM, False, [tuple(beam_start), tuple(beam_end)], LINE_WIDTH_SENSOR_BEAM)        
        if False:
            # drawing area for initial random positions
            tl = np.array([5.0, 5.0])
            tr = tl + np.array([10.0, 0.0])
            bl = tl + np.array([0.0, -10.0])
            br = bl + np.array([10.0, 0.0])        
            initial_area = [(tl, tr), (tr, br), (br, bl), (bl, tl)]    
            for start, end in initial_area:
                seg_start = SCALER_A * start + SCALER_B
                seg_end = SCALER_A * end + SCALER_B        
                draw_dashed_line(screen, COLOR_INITIAL_AREA_AND_ANGLE, tuple(seg_start), tuple(seg_end), 4)            
        # drawing "parked" label                
        if car.parked_:
            font = pygame.font.SysFont(TEXT_FONT_NAME, TEXT_MESSAGE_FONT_SIZE) 
            text_img = font.render("PARKED!", True, COLOR_PARKED)
            text_rect = text_img.get_rect(center=(SCREEN_RESOLUTION[0] // 2, SCREEN_RESOLUTION[1] // 2))
            screen.blit(text_img, text_rect)                        
    else:
        # drawing collision
        pygame.draw.circle(screen, COLOR_COLLISION, SCALER_A * car.collision_x_ + SCALER_B, RADIUS_COLLISION, LINE_WIDTH_COLLISION)
        font = pygame.font.SysFont(TEXT_FONT_NAME, TEXT_MESSAGE_FONT_SIZE)
        text_img = font.render("COLLISION!", True, COLOR_COLLISION)
        text_rect = text_img.get_rect(center=(SCREEN_RESOLUTION[0] // 2, SCREEN_RESOLUTION[1] // 2))
        screen.blit(text_img, text_rect)        
    # printouts
    float_format = "{:+06.2f}".format
    time_format = "{:+08.3f}".format
    font = pygame.font.SysFont(TEXT_FONT_NAME, TEXT_FONT_SIZE) 
    # printing Q values around car
    if Q_pred is not None:
        dist_factor = 5.0
        am = np.argmax(Q_pred)
        for i, (ap, q) in enumerate(zip(ACTION_PAIRS, Q_pred)):
            qx = car.x_ + ap[0] * car.d_ahead_ * dist_factor + ap[1] * car.d_right_ * dist_factor
            qxs = SCALER_A * qx + SCALER_B
            q_color = COLOR_TEXT_HIGHLIGHT if am == i else COLOR_TEXT
            text_img = font.render(f"{float_format(q)}", True, q_color)
            screen.blit(text_img, (qxs[0] - 1.5 * TEXT_FONT_SIZE, qxs[1]))    
    # printing time
    text_img = font.render(f"t: {time_format(time_elapsed)} s", True, COLOR_TEXT)
    screen.blit(text_img, (TEXT_MARGIN, TEXT_MARGIN))    
    np.set_printoptions(formatter={"float_kind" : float_format})
    # printing physics of car                
    text_img = font.render(f"x: {car.x_} m", True, COLOR_TEXT)
    screen.blit(text_img, (TEXT_MARGIN, TEXT_MARGIN + 1 * TEXT_FONT_SIZE))
    text_img = font.render(f"d ahead: {car.d_ahead_} m, angle ahead: {float_format(car.angle_ahead_)}", True, COLOR_TEXT)    
    screen.blit(text_img, (TEXT_MARGIN, TEXT_MARGIN + 2 * TEXT_FONT_SIZE))    
    text_img = font.render(f"v: {car.v_} m/s, |v|: {float_format(car.v_magnitude_)} m/s = {float_format(car.v_magnitude_ * 3.6)} km/h", True, COLOR_TEXT)
    screen.blit(text_img, (TEXT_MARGIN, TEXT_MARGIN + 3 * TEXT_FONT_SIZE))
    text_img = font.render(f"a: {car.a_} m/s^2, |a|: {float_format(car.a_magnitude_)} m/s^2", True, COLOR_TEXT)
    screen.blit(text_img, (TEXT_MARGIN, TEXT_MARGIN + 4 * TEXT_FONT_SIZE))
    # printing to park place info
    text_img = font.render(f"to park place f: {car.to_park_place_f_} m", True, COLOR_TEXT)
    screen.blit(text_img, (TEXT_MARGIN, TEXT_MARGIN + 5 * TEXT_FONT_SIZE))
    text_img = font.render(f"to park place b: {car.to_park_place_b_} m", True, COLOR_TEXT)
    screen.blit(text_img, (TEXT_MARGIN, TEXT_MARGIN + 6 * TEXT_FONT_SIZE))    
    text_img = font.render(f"to park place fl: {car.to_park_place_fl_} m, fl2: {car.to_park_place_fl2_}", True, COLOR_TEXT)
    screen.blit(text_img, (TEXT_MARGIN, TEXT_MARGIN + 7 * TEXT_FONT_SIZE))
    text_img = font.render(f"to park place fr: {car.to_park_place_fr_} m, fr2: {car.to_park_place_fr2_} m", True, COLOR_TEXT)
    screen.blit(text_img, (TEXT_MARGIN, TEXT_MARGIN + 8 * TEXT_FONT_SIZE))
    text_img = font.render(f"to park place bl: {car.to_park_place_bl_} m, bl2: {car.to_park_place_bl2_} m", True, COLOR_TEXT)
    screen.blit(text_img, (TEXT_MARGIN, TEXT_MARGIN + 9 * TEXT_FONT_SIZE))
    text_img = font.render(f"to park place br: {car.to_park_place_br_} m, br2: {car.to_park_place_br2_} m", True, COLOR_TEXT)
    screen.blit(text_img, (TEXT_MARGIN, TEXT_MARGIN + 10 * TEXT_FONT_SIZE))    
    text_img = font.render(f"distance: {float_format(car.distance_)} m", True, COLOR_TEXT)
    screen.blit(text_img, (TEXT_MARGIN, TEXT_MARGIN + 11 * TEXT_FONT_SIZE))
    text_img = font.render(f"angle distance: {float_format(car.angle_distance_)}", True, COLOR_TEXT)
    screen.blit(text_img, (TEXT_MARGIN, TEXT_MARGIN + 12 * TEXT_FONT_SIZE))
    text_img = font.render(f"gutter distance: {float_format(car.gutter_distance_)} m", True, COLOR_TEXT)
    screen.blit(text_img, (TEXT_MARGIN, TEXT_MARGIN + 13 * TEXT_FONT_SIZE))                            
    # printing car sensors' state
    text_img = font.render(f"sensors f: {car.sensors_front_values_} m", True, COLOR_TEXT)
    screen.blit(text_img, (TEXT_MARGIN, TEXT_MARGIN + 14 * TEXT_FONT_SIZE))
    text_img = font.render(f"sensors b: {car.sensors_back_values_} m", True, COLOR_TEXT)
    screen.blit(text_img, (TEXT_MARGIN, TEXT_MARGIN + 15 * TEXT_FONT_SIZE))    
    text_img = font.render(f"sensors l: {car.sensors_left_values_} m", True, COLOR_TEXT)
    screen.blit(text_img, (TEXT_MARGIN, TEXT_MARGIN + 16 * TEXT_FONT_SIZE))
    text_img = font.render(f"sensors r: {car.sensors_right_values_} m", True, COLOR_TEXT)
    screen.blit(text_img, (TEXT_MARGIN, TEXT_MARGIN + 17 * TEXT_FONT_SIZE))    
    # printing reward    
    text_img = font.render(f"reward: {float_format(car.reward_)}", True, COLOR_TEXT)
    screen.blit(text_img, (TEXT_MARGIN, TEXT_MARGIN + 18 * TEXT_FONT_SIZE))          
    # episode time limit check
    if time_elapsed >= QL_EPISODE_TIME_LIMIT and not car.parked_:
        font = pygame.font.SysFont(TEXT_FONT_NAME, TEXT_MESSAGE_FONT_SIZE) 
        text_img = font.render("TIME LIMIT EXCEEDED", True, COLOR_TIME_LIMIT_EXCEEDED)
        text_rect = text_img.get_rect(center=(SCREEN_RESOLUTION[0] // 2, SCREEN_RESOLUTION[1] // 2))
        screen.blit(text_img, text_rect)    

def scene_pp_west_side_10_angle_halfpi():
    west_x = -10.0
    east_x = 10.0
    side = 10.0
    angle_range = 0.5 * np.pi
    ppfl = np.array([west_x - 0.5 * PARK_PLACE_LENGTH, -0.5 * PARK_PLACE_WIDTH])    
    ppfr = ppfl + np.array([0.0, PARK_PLACE_WIDTH])
    park_place = ParkPlace(ppfl, ppfr, ppfl + np.array([PARK_PLACE_LENGTH, 0.0]), ppfr + np.array([PARK_PLACE_LENGTH, 0.0]))    
    random_shift = np.array([(2.0 * np.random.rand() - 1.0) * 0.5 * side, (2.0 * np.random.rand() - 1.0) * 0.5 * side])    
    random_angle = (2.0 * np.random.rand() - 1.0) * 0.5 * angle_range    
    car = Car(x=np.array([east_x, 0.0]) + random_shift, angle=0.5 * np.pi + random_angle)    
    obstacles = []        
    scene = Scene(car, park_place, obstacles)
    return scene

def scene_pp_west_side_10_angle_pi():
    west_x = -10.0
    east_x = 10.0
    side = 10.0
    angle_range = 1.0 * np.pi
    ppfl = np.array([west_x - 0.5 * PARK_PLACE_LENGTH, -0.5 * PARK_PLACE_WIDTH])    
    ppfr = ppfl + np.array([0.0, PARK_PLACE_WIDTH])
    park_place = ParkPlace(ppfl, ppfr, ppfl + np.array([PARK_PLACE_LENGTH, 0.0]), ppfr + np.array([PARK_PLACE_LENGTH, 0.0]))    
    random_shift = np.array([(2.0 * np.random.rand() - 1.0) * 0.5 * side, (2.0 * np.random.rand() - 1.0) * 0.5 * side])    
    random_angle = (2.0 * np.random.rand() - 1.0) * 0.5 * angle_range    
    car = Car(x=np.array([east_x, 0.0]) + random_shift, angle=0.5 * np.pi + random_angle)    
    obstacles = []        
    scene = Scene(car, park_place, obstacles)
    return scene

def scene_pp_westeast_side_10_angle_pi():
    west_x = -10.0
    east_x = 10.0    
    side = 10.0
    angle_range = 1.0 * np.pi    
    random_shift = np.array([(2.0 * np.random.rand() - 1.0) * 0.5 * side, (2.0 * np.random.rand() - 1.0) * 0.5 * side])    
    random_angle = (2.0 * np.random.rand() - 1.0) * 0.5 * angle_range    
    if np.random.rand() < 0.5:
        ppfl = np.array([west_x - 0.5 * PARK_PLACE_LENGTH, -0.5 * PARK_PLACE_WIDTH])    
        ppfr = ppfl + np.array([0.0, PARK_PLACE_WIDTH])
        park_place = ParkPlace(ppfl, ppfr, ppfl + np.array([PARK_PLACE_LENGTH, 0.0]), ppfr + np.array([PARK_PLACE_LENGTH, 0.0]))        
        car = Car(x=np.array([east_x, 0.0]) + random_shift, angle=0.5 * np.pi + random_angle)    
        scene = Scene(car, park_place, obstacles=[])
    else:    
        ppfl = np.array([east_x + 0.5 * PARK_PLACE_LENGTH, 0.5 * PARK_PLACE_WIDTH])    
        ppfr = ppfl + np.array([0.0, -PARK_PLACE_WIDTH])
        park_place = ParkPlace(ppfl, ppfr, ppfl + np.array([-PARK_PLACE_LENGTH, 0.0]), ppfr + np.array([-PARK_PLACE_LENGTH, 0.0]))
        car = Car(x=np.array([west_x, 0.0]) + random_shift, angle=-(0.5 * np.pi + random_angle))
        scene = Scene(car, park_place, obstacles=[])        
    return scene

def scene_pp_middle_side_20_angle_twopi():
    side = 20.0
    angle_range = 2.0 * np.pi        
    ppfl = np.array([0.0 - 0.5 * PARK_PLACE_LENGTH, -0.5 * PARK_PLACE_WIDTH])
    ppfr = ppfl + np.array([0.0, PARK_PLACE_WIDTH])
    park_place = ParkPlace(ppfl, ppfr, ppfl + np.array([PARK_PLACE_LENGTH, 0.0]), ppfr + np.array([PARK_PLACE_LENGTH, 0.0]))
    random_shift = np.array([(2.0 * np.random.rand() - 1.0) * 0.5 * side, (2.0 * np.random.rand() - 1.0) * 0.5 * side])
    random_angle = (2.0 * np.random.rand() - 1.0) * 0.5 * angle_range
    middle_x = np.array([0.0, 0.0]) 
    car = Car(x=middle_x + random_shift, angle=0.5 * np.pi + random_angle)
    obstacles = []
    scene = Scene(car, park_place, obstacles)
    return scene

def scene_pp_random_car_random_side_20():
    pp_side = 20.0
    pp_angle_range = 2.0 * np.pi
    car_side = 20.0
    car_angle_range = 2.0 * np.pi            
    ppfl = np.array([0.0 - 0.5 * PARK_PLACE_LENGTH, -0.5 * PARK_PLACE_WIDTH])
    ppfr = ppfl + np.array([0.0, PARK_PLACE_WIDTH])
    ppbl = ppfl + np.array([PARK_PLACE_LENGTH, 0.0])
    ppbr = ppfr + np.array([PARK_PLACE_LENGTH, 0.0])
    random_angle = (2 * np.random.rand() - 1) * 0.5 * pp_angle_range
    rotation_matrix = np.array([[np.cos(random_angle), -np.sin(random_angle)], [np.sin(random_angle), np.cos(random_angle)]])
    pp_random_shift = np.array([(2 * np.random.rand() - 1) * 0.5 * pp_side, (2 * np.random.rand() - 1) * 0.5 * pp_side])
    ppfl = rotation_matrix.dot(ppfl) + pp_random_shift
    ppfr = rotation_matrix.dot(ppfr) + pp_random_shift
    ppbl = rotation_matrix.dot(ppbl) + pp_random_shift
    ppbr = rotation_matrix.dot(ppbr) + pp_random_shift
    park_place = ParkPlace(ppfl, ppfr, ppbl, ppbr)
    car_random_shift = np.array([(2 * np.random.rand() - 1) * 0.5 * car_side, (2 * np.random.rand() - 1) * 0.5 * car_side])
    car_random_angle = (2 * np.random.rand() - 1) * 0.5 * car_angle_range
    car = Car(x=np.array([0.0, 0.0]) + car_random_shift, angle=0.5 * np.pi + car_random_angle)
    obstacles = []
    scene = Scene(car, park_place, obstacles)
    return scene

def scene_obstacles_1():
    car_side = 10.0
    car_angle_range = 0.5 * np.pi   
    ppfl = np.array([-14.0, -6.0])    
    ppfr = ppfl + np.array([0.0, PARK_PLACE_WIDTH])
    park_place = ParkPlace(ppfl, ppfr, ppfl + np.array([PARK_PLACE_LENGTH, 0.0]), ppfr + np.array([PARK_PLACE_LENGTH, 0.0]))
    obstacles = [
        #Obstacle([ppfr + np.array([0.0, 1.0]), ppfr + np.array([0.0, 1.0 + 2 * PARK_PLACE_WIDTH]), ppfr + np.array([PARK_PLACE_LENGTH * 1.5, 1.0 + 2 * PARK_PLACE_WIDTH]), ppfr + np.array([PARK_PLACE_LENGTH * 1.5, 1.0])]),
        Obstacle([ppfl + np.array([0.0, -1.0]), ppfl + np.array([0.0, -1.0 - 2 * PARK_PLACE_WIDTH]), ppfl + np.array([PARK_PLACE_LENGTH * 1.5, -1.0 - 2 * PARK_PLACE_WIDTH]), ppfl + np.array([PARK_PLACE_LENGTH * 1.5, -1.0])])
        ]    
    car_random_shift = np.array([(2 * np.random.rand() - 1) * 0.5 * car_side, (2 * np.random.rand() - 1) * 0.5 * car_side])
    car_random_angle = (2 * np.random.rand() - 1) * 0.5 * car_angle_range
    car = Car(x=np.array([10.0, 0.0]) + car_random_shift, angle=0.5 * np.pi + car_random_angle)
    scene = Scene(car, park_place, obstacles)
    return scene

def scene_obstacles_and_pole_angle_halfpi():
    west_x = -12.5
    east_x = 12.5
    side = 10.0
    angle_range = 0.5 * np.pi
    ppfl = np.array([west_x - 0.5 * PARK_PLACE_LENGTH, -0.5 * PARK_PLACE_WIDTH])    
    ppfr = ppfl + np.array([0.0, PARK_PLACE_WIDTH])
    park_place = ParkPlace(ppfl, ppfr, ppfl + np.array([PARK_PLACE_LENGTH, 0.0]), ppfr + np.array([PARK_PLACE_LENGTH, 0.0]))    
    random_shift = np.array([(2.0 * np.random.rand() - 1.0) * 0.5 * side, (2.0 * np.random.rand() - 1.0) * 0.5 * side]) 
    random_angle = (2.0 * np.random.rand() - 1.0) * 0.5 * angle_range    
    car = Car(x=np.array([east_x, 0.0]) + random_shift, angle=0.5 * np.pi + random_angle)
    o_pp_dist = 1.0 
    o_shift = np.array([0.0, 3 * PARK_PLACE_WIDTH + 2 * o_pp_dist])
    o_central_radius = 1.0
    obstacles = [
        Obstacle([ppfl + np.array([0.0, -o_pp_dist]), ppfl + np.array([0.0, -o_pp_dist - 2 * PARK_PLACE_WIDTH]), 
                  ppfl + np.array([PARK_PLACE_LENGTH, -o_pp_dist - 2 * PARK_PLACE_WIDTH]), ppfl + np.array([PARK_PLACE_LENGTH, -o_pp_dist])]),
        Obstacle([ppfl + np.array([0.0, -o_pp_dist]) + o_shift, ppfl + np.array([0.0, -o_pp_dist - 2 * PARK_PLACE_WIDTH]) + o_shift, 
                  ppfl + np.array([PARK_PLACE_LENGTH, -o_pp_dist - 2 * PARK_PLACE_WIDTH]) + o_shift, ppfl + np.array([PARK_PLACE_LENGTH, -o_pp_dist]) + o_shift]),
        Obstacle([np.array([-o_central_radius, o_central_radius]), np.array([o_central_radius, o_central_radius]), 
                  np.array([o_central_radius, -o_central_radius]), np.array([-o_central_radius, -o_central_radius])])                        
        ]    
    scene = Scene(car, park_place, obstacles)
    return scene

def scene_pp_middle_obstacles_car_westeast_side_10_angle_twopi():
    side = 10.0
    angle_range = 2.0 * np.pi
    ppfl = np.array([0.0 - 0.5 * PARK_PLACE_LENGTH, -0.5 * PARK_PLACE_WIDTH])
    ppfr = ppfl + np.array([0.0, PARK_PLACE_WIDTH])
    park_place = ParkPlace(ppfl, ppfr, ppfl + np.array([PARK_PLACE_LENGTH, 0.0]), ppfr + np.array([PARK_PLACE_LENGTH, 0.0]))
    random_shifts = [np.array([PARK_PLACE_LENGTH + 0.5 * side + (2.0 * np.random.rand() - 1.0) * 0.5 * side, (2.0 * np.random.rand() - 1.0) * 0.5 * side]),
                     np.array([-PARK_PLACE_LENGTH - 0.5 * side + (2.0 * np.random.rand() - 1.0) * 0.5 * side, (2.0 * np.random.rand() - 1.0) * 0.5 * side])]                        
    random_shift = random_shifts[np.random.randint(2)]
    random_angle = (2.0 * np.random.rand() - 1.0) * 0.5 * angle_range
    middle_x = np.array([0.0, 0.0]) 
    car = Car(x=middle_x + random_shift, angle=0.5 * np.pi + random_angle)
    o_pp_dist = 1.0 
    o_shift = np.array([0.0, 2 * PARK_PLACE_WIDTH + 2 * o_pp_dist])
    obstacles = [
        Obstacle([ppfl + np.array([0.0, -o_pp_dist]), ppfl + np.array([0.0, -o_pp_dist - 1 * PARK_PLACE_WIDTH]), 
                  ppfl + np.array([PARK_PLACE_LENGTH, -o_pp_dist - 1 * PARK_PLACE_WIDTH]), ppfl + np.array([PARK_PLACE_LENGTH, -o_pp_dist])]),
        Obstacle([ppfl + np.array([0.0, -o_pp_dist]) + o_shift, ppfl + np.array([0.0, -o_pp_dist - 1 * PARK_PLACE_WIDTH]) + o_shift, 
                  ppfl + np.array([PARK_PLACE_LENGTH, -o_pp_dist - 1 * PARK_PLACE_WIDTH]) + o_shift, ppfl + np.array([PARK_PLACE_LENGTH, -o_pp_dist]) + o_shift])                        
        ]    

    scene = Scene(car, park_place, obstacles)
    return scene

def scene_pp_west_obstacles_side_10_angle_pi():
    west_x = -10.0
    east_x = 10.0
    side = 10.0
    angle_range = 1.0 * np.pi
    ppfl = np.array([west_x - 0.5 * PARK_PLACE_LENGTH, -0.5 * PARK_PLACE_WIDTH])    
    ppfr = ppfl + np.array([0.0, PARK_PLACE_WIDTH])
    park_place = ParkPlace(ppfl, ppfr, ppfl + np.array([PARK_PLACE_LENGTH, 0.0]), ppfr + np.array([PARK_PLACE_LENGTH, 0.0]))    
    random_shift = np.array([(2.0 * np.random.rand() - 1.0) * 0.5 * side, (2.0 * np.random.rand() - 1.0) * 0.5 * side])    
    random_angle = (2.0 * np.random.rand() - 1.0) * 0.5 * angle_range    
    car = Car(x=np.array([east_x, 0.0]) + random_shift, angle=0.5 * np.pi + random_angle)
    o_pp_dist = 1.0 
    o_shift = np.array([0.0, 2 * PARK_PLACE_WIDTH + 2 * o_pp_dist])    
    obstacles = [
        Obstacle([ppfl + np.array([0.0, -o_pp_dist]), ppfl + np.array([0.0, -o_pp_dist - 1 * PARK_PLACE_WIDTH]), 
                  ppfl + np.array([PARK_PLACE_LENGTH, -o_pp_dist - 1 * PARK_PLACE_WIDTH]), ppfl + np.array([PARK_PLACE_LENGTH, -o_pp_dist])]),
        Obstacle([ppfl + np.array([0.0, -o_pp_dist]) + o_shift, ppfl + np.array([0.0, -o_pp_dist - 1 * PARK_PLACE_WIDTH]) + o_shift, 
                  ppfl + np.array([PARK_PLACE_LENGTH, -o_pp_dist - 1 * PARK_PLACE_WIDTH]) + o_shift, ppfl + np.array([PARK_PLACE_LENGTH, -o_pp_dist]) + o_shift])                        
        ]    
    scene = Scene(car, park_place, obstacles)
    return scene

def scene_pp_middle_obstacles_oppdist_1_side_10_angle_pi():
    west_x = 0.0
    east_x = 10.0
    side = 10.0
    angle_range = 1.0 * np.pi
    ppfl = np.array([west_x - 0.5 * PARK_PLACE_LENGTH, -0.5 * PARK_PLACE_WIDTH])    
    ppfr = ppfl + np.array([0.0, PARK_PLACE_WIDTH])
    park_place = ParkPlace(ppfl, ppfr, ppfl + np.array([PARK_PLACE_LENGTH, 0.0]), ppfr + np.array([PARK_PLACE_LENGTH, 0.0]))    
    random_shift = np.array([(2.0 * np.random.rand() - 1.0) * 0.5 * side, (2.0 * np.random.rand() - 1.0) * 0.5 * side])    
    random_angle = (2.0 * np.random.rand() - 1.0) * 0.5 * angle_range    
    car = Car(x=np.array([east_x, 0.0]) + random_shift, angle=0.5 * np.pi + random_angle)
    o_pp_dist = 1.0 
    o_shift = np.array([0.0, 2 * PARK_PLACE_WIDTH + 2 * o_pp_dist])    
    obstacles = [
        Obstacle([ppfl + np.array([0.0, -o_pp_dist]), ppfl + np.array([0.0, -o_pp_dist - 1 * PARK_PLACE_WIDTH]), 
                  ppfl + np.array([PARK_PLACE_LENGTH, -o_pp_dist - 1 * PARK_PLACE_WIDTH]), ppfl + np.array([PARK_PLACE_LENGTH, -o_pp_dist])]),
        Obstacle([ppfl + np.array([0.0, -o_pp_dist]) + o_shift, ppfl + np.array([0.0, -o_pp_dist - 1 * PARK_PLACE_WIDTH]) + o_shift, 
                  ppfl + np.array([PARK_PLACE_LENGTH, -o_pp_dist - 1 * PARK_PLACE_WIDTH]) + o_shift, ppfl + np.array([PARK_PLACE_LENGTH, -o_pp_dist]) + o_shift])                        
        ]    
    scene = Scene(car, park_place, obstacles)
    return scene

def scene_pp_middle_obstacles_oppdist_05_side_10_angle_pi():
    west_x = 0.0
    east_x = 10.0
    side = 10.0
    angle_range = 1.0 * np.pi
    ppfl = np.array([west_x - 0.5 * PARK_PLACE_LENGTH, -0.5 * PARK_PLACE_WIDTH])    
    ppfr = ppfl + np.array([0.0, PARK_PLACE_WIDTH])
    park_place = ParkPlace(ppfl, ppfr, ppfl + np.array([PARK_PLACE_LENGTH, 0.0]), ppfr + np.array([PARK_PLACE_LENGTH, 0.0]))    
    random_shift = np.array([(2.0 * np.random.rand() - 1.0) * 0.5 * side, (2.0 * np.random.rand() - 1.0) * 0.5 * side])    
    random_angle = (2.0 * np.random.rand() - 1.0) * 0.5 * angle_range    
    car = Car(x=np.array([east_x, 0.0]) + random_shift, angle=0.5 * np.pi + random_angle)
    o_pp_dist = 0.5 
    o_shift = np.array([0.0, 2 * PARK_PLACE_WIDTH + 2 * o_pp_dist])    
    obstacles = [
        Obstacle([ppfl + np.array([0.0, -o_pp_dist]), ppfl + np.array([0.0, -o_pp_dist - 1 * PARK_PLACE_WIDTH]), 
                  ppfl + np.array([PARK_PLACE_LENGTH, -o_pp_dist - 1 * PARK_PLACE_WIDTH]), ppfl + np.array([PARK_PLACE_LENGTH, -o_pp_dist])]),
        Obstacle([ppfl + np.array([0.0, -o_pp_dist]) + o_shift, ppfl + np.array([0.0, -o_pp_dist - 1 * PARK_PLACE_WIDTH]) + o_shift, 
                  ppfl + np.array([PARK_PLACE_LENGTH, -o_pp_dist - 1 * PARK_PLACE_WIDTH]) + o_shift, ppfl + np.array([PARK_PLACE_LENGTH, -o_pp_dist]) + o_shift])                        
        ]    
    scene = Scene(car, park_place, obstacles)
    return scene 

# MAIN
if __name__ == "__main__":
            
    print("CAR PARKING EXPERIMENT...")
    ehs = experiment_hash_str() 
    print(f"EXPERIMENT HASH: {ehs}") 
    if not LEARNING_ON and TEST_SCENE_FUNCTION_NAME == None:
        print(f"EXPERIMENT MODE: " + ("LEARNING" if LEARNING_ON else f"TESTING [test model name: {TEST_MODEL_NAME}]"))
    else:
        print(f"EXPERIMENT MODE: " + ("LEARNING" if LEARNING_ON else f"TESTING [test model name: {TEST_MODEL_NAME}, test scene function name: {TEST_SCENE_FUNCTION_NAME}]"))
    print(f"EXPERIMENT PARAMETERS:\n {dict_to_str(experiment_params())}")
    
    if SCREEN_RESOLUTION_DPI_AWARE:
        ctypes.windll.user32.SetProcessDPIAware()
    
    t1_main = time.time()    
    if LEARNING_ON:
        np.random.seed(QL_RANDOM_SEED)
        n_episodes = QL_N_EPISODES        
        animation_on = QL_ANIMATION_ON
        animation_frequency = QL_ANIMATION_FREQUENCY                
    else:
        np.random.seed(TEST_RANDOM_SEED)
        n_episodes = TEST_N_EPISODES
        animation_on = TEST_ANIMATION_ON
        animation_frequency = 1         
    seed_dtype = np.int32    
    epi_seeds = np.random.randint(low=0, high=np.iinfo(seed_dtype).max, dtype=seed_dtype, size=n_episodes)
    if not LEARNING_ON and TEST_EPI_SEEDS:
        epi_seeds = np.array(TEST_EPI_SEEDS)
        n_episodes = epi_seeds.size    
    scene_function_name = QL_SCENE_FUNCTION_NAME 
    if not LEARNING_ON and TEST_SCENE_FUNCTION_NAME is not None: 
        scene_function_name = TEST_SCENE_FUNCTION_NAME
    scene_function = globals()["scene_" + scene_function_name]            
    first_fit_done = False
    Q = None 
    Q_oracle = None    
    parked_count = 0.0
    parked_frequency = 0.0
    parked_frequency_ema = 0.0
    rewards_ema = 0.0
    distances_ema = 0.0    
    r2_ema = 0.0    
    extras = {"parked_count": [], 
              "parked_frequency": [], 
              "parked_frequency_ema": [], 
              "rewards_ema": [], 
              "distances_ema": [], 
              "mse_batch_before": [], 
              "mse_batch_after": [], 
              "r2_batch_before": [], 
              "r2_batch_after": []} 
    eb = np.empty((EXPERIENCE_BUFFER_MAX_SIZE, 6), dtype=object) # state, action, reward, next state, is next state terminal, Bellman error
    eb_size = 0
    eb_size_old = 0
    if not LEARNING_ON and TEST_MODEL_NAME:
        [Q] = unpickle_all(FOLDER_MODELS + TEST_MODEL_NAME + "_q.bin")  
    elif LEARNING_ON and QL_INITIAL_MODEL_NAME:
        [Q] = unpickle_all(FOLDER_MODELS + QL_INITIAL_MODEL_NAME + "_q.bin")
        first_fit_done = True
        Q_oracle = deepcopy(Q)

    eps = QL_EPS_MAX    
    scene = scene_function()
    state = scene.car_.get_state() # fake state to 'warm up' transformer
    QL_TRANSFORMER.fit_transform(np.array([state]))
    n = QL_TRANSFORMER.n_output_features_    
    print(f"FEATURES IN STATE REPRESENTATION: {n}")
    epi_disp_separator = "-" * 256
    dt_since_action = QL_DT * QL_STEERING_GAP_STEPS

    # INTRO
    if animation_on:
        pygame.init()
        icon = pygame.image.load("./../img/icon.png")
        pygame.display.set_icon(icon)
        pygame.display.set_caption("CAR PARKING Q-LEARNING DEMO")
        screen = pygame.display.set_mode(SCREEN_RESOLUTION)
    if animation_on and DEMO_TITLE_LINE_1 is not None:    
        draw_intro(screen, ehs, DEMO_TITLE_LINE_1, DEMO_TITLE_LINE_2)
        pygame.display.flip()
        input("[press enter to start]")
    
    # LOOP OVER EPISODES
    for epi in range(n_episodes):        
        t1_loop_body = time.time()
        epi_seed = epi_seeds[epi]
        np.random.seed(epi_seed)
        epi_title = f"CAR PARKING Q-LEARNING, EPISODE: {epi + 1}/{n_episodes}... " + (f"[epsilon: {eps}, seed: {epi_seed}]" if LEARNING_ON else f"[seed: {epi_seed}]")             
        epi_animate = False
        scene = scene_function()
        if animation_on and epi % animation_frequency == 0:
            pygame.init()
            icon = pygame.image.load("./../img/icon.png")    
            pygame.display.set_icon(icon)
            screen = pygame.display.set_mode(SCREEN_RESOLUTION)    
            clock = pygame.time.Clock()            
            pygame.display.set_caption(epi_title)
            epi_animate = True
        manual_steering = epi_animate and not (LEARNING_ON or TEST_MODEL_NAME)
        print(epi_disp_separator + "\n" + epi_title)
        if epi_animate:
            print(f"[animating this episode...]")        
        car = scene.car_
        state = car.get_state()
        next_state = None
        QL_TRANSFORMER.fit_transform(np.array([state]))
        t1 = time.time()
        t2 = None
        time_elapsed = 0.0
        frame = 0
        rewards_total = 0.0
        rewards_count = 0        
        distances_total = 0.0
        epi_eb = np.empty((int(2 * QL_EPISODE_TIME_LIMIT / QL_DT), 6), dtype=object)        
        epi_eb_size = 0        
        collect_next_experience = False
        collect_next_reward = False
        antistuck_nudge_ongoing = False
        antistuck_nudge_count = 0
        while True: # loop (simulation) for current episode
            time_elapsed = frame * QL_DT
            time_remaining = QL_EPISODE_TIME_LIMIT - time_elapsed                                        
            epi_stop_condition = car.parked_ or car.collided_ or time_elapsed >= QL_EPISODE_TIME_LIMIT                                                     
            steering_now = frame % QL_STEERING_GAP_STEPS == 0
            if LEARNING_ON:
                if (steering_now and collect_next_experience) or epi_stop_condition:
                    next_state = car.get_state()
                    action = ACTION_PAIRS_INDEXER[tuple(action_pair)]
                    terminal = car.parked_ or car.collided_
                    experience = np.empty((1, 6), dtype=object)
                    experience[0, 0] = state
                    experience[0, 1] = action
                    experience[0, 2] = car.reward_
                    experience[0, 3] = next_state                    
                    experience[0, 4] = terminal 
                    experience[0, 5] = 0.0 # Bellman error (for the case of prioritized experience replay, updataeble after fits)                   
                    epi_eb[epi_eb_size] = experience
                    epi_eb_size += 1
                    if False and terminal: # additional experiences (currently off) for terminal states -> no matter what action taken in them, the next state and reward stay the same (for targets preparation purposes in ML)
                        for a in range(len(ACTION_PAIRS)):
                            experience = np.empty((1, 6), dtype=object)
                            experience[0, 0] = next_state
                            experience[0, 1] = a
                            experience[0, 2] = car.reward_
                            experience[0, 3] = next_state                    
                            experience[0, 4] = terminal 
                            experience[0, 5] = 0.0 # Bellman error (for the case of prioritized experience replay, updataeble after fits)                   
                            epi_eb[epi_eb_size] = experience
                            epi_eb_size += 1                                                    
            if (steering_now and collect_next_reward) or epi_stop_condition:                        
                rewards_total += car.reward_
                rewards_count += 1                
            if epi_stop_condition:
                if epi_animate:
                    draw_scene(screen, scene, time_elapsed, None)
                    pygame.display.flip()                     
                    time.sleep(1.0)
                    pygame.quit()
                t2 = time.time()
                if car.parked_:
                    parked_count += 1                    
                parked_frequency = parked_count / (epi + 1)
                break
            if steering_now and not antistuck_nudge_ongoing:
                state = car.get_state()
                collect_next_reward = True
                if LEARNING_ON:
                    collect_next_experience = np.random.rand() < QL_COLLECT_EXPERIENCE_PROBABILITY
                if Q is None:
                    Q_pred = None
                else:
                    X_state = QL_TRANSFORMER.fit_transform(np.array([state]))
                    Q_pred = Q.predict(X_state)[0]             
            if epi_animate:
                clock.tick(QL_FPS)                                    
                # handling UI events
                while_break = False
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        sys.exit(0)
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        while_break = True
                if while_break:
                    break         
                if steering_now and manual_steering:                   
                    keys = pygame.key.get_pressed()
                    action_pair = [0, 0] # first: ahead or back, second: right or left 
                    if keys[pygame.K_UP]:
                        action_pair[0] = 1
                    elif keys[pygame.K_DOWN]:
                        action_pair[0] = -1
                    if keys[pygame.K_RIGHT]:
                        action_pair[1] = 1                
                    elif keys[pygame.K_LEFT]:
                        action_pair[1] = -1                                
            if steering_now and not manual_steering:
                if antistuck_nudge_ongoing:
                    antistuck_nudge_steering_steps -= 1
                    if antistuck_nudge_steering_steps == 0:
                        antistuck_nudge_ongoing = False
                else:   
                    action_index = np.random.choice(len(ACTION_PAIRS)) # random action
                    if Q_pred is not None and ((LEARNING_ON and np.random.rand() <= 1.0 - eps) or ((not LEARNING_ON and np.random.rand() <= 1.0 - TEST_EPS) or epi_animate)):                    
                        action_index = np.argmax(Q_pred) # greedy action
                    action_pair = ACTION_PAIRS[action_index]                                                                                                                                                              
            # applying action (Q-driven or manual) from such last step where steering took place 
            car.accelerations_imposed_ = []
            if action_pair[0] > 0:
                car.accelerate_ahead(ACCELERATION_MAGNITUDES_AHEAD[action_pair[0]])
            elif action_pair[0] < 0:
                car.accelerate_back(ACCELERATION_MAGNITUDES_BACK[-action_pair[0]])            
            if action_pair[1] > 0:
                car.accelerate_right(ACCELERATION_MAGNITUDES_SIDE[action_pair[1]])
            elif action_pair[1] < 0:
                car.accelerate_left(ACCELERATION_MAGNITUDES_SIDE[-action_pair[1]])                
            if epi_animate:  
                draw_scene(screen, scene, time_elapsed, Q_pred)
                pygame.display.flip()                                         
            if QL_ANTISTUCK_NUDGE and steering_now and not manual_steering and not antistuck_nudge_ongoing: # random nudge (if not yet parked, and not about to move, do random acceleration ahead or back)                        
                if car.v_magnitude_ == 0.0 and car.a_magnitude_ == 0.0 or car.is_stuck(QL_DT):
                    non_side_acceleration_indexes = np.array([ACTION_PAIRS_INDEXER[(-1, 0)], ACTION_PAIRS_INDEXER[(1, 0)]])
                    action_index = np.random.choice(non_side_acceleration_indexes)
                    action_pair = ACTION_PAIRS[action_index]
                    antistuck_nudge_ongoing = True
                    antistuck_nudge_steering_steps = QL_ANTISTUCK_NUDGE_STEERING_STEPS
                    antistuck_nudge_count += 1        
                    # print(f"[antistuck nudge: {antistuck_nudge_count}]") 
            car.step(QL_DT, dt_since_action, time_remaining, scene.obstacles_, scene.park_place_)                                                                      
            distances_total += car.distance_                        
            frame += 1            
            t2 = time.time()            
            time_elapsed = t2 - t1                           
        epi_outcome_str = "time_exceeded"
        if car.collided_:
            epi_outcome_str = "collision"
        elif car.parked_:
            epi_outcome_str= "parked"
        max_frames = QL_EPISODE_TIME_LIMIT / QL_DT
        fps_observed = 0.0
        if t2 - t1 > 0.0:
            fps_observed = frame / (t2 - t1) 
        print(f"CAR PARKING Q-LEARNING, EPISODE: {epi + 1}/{n_episodes} DONE. [outcome: {epi_outcome_str}, frames performed: {frame}, last reward: {car.reward_}, mean reward: {rewards_total / rewards_count}, mean distance: {distances_total / max_frames}, time: {t2 - t1} s, fps: {fps_observed}]")
        # appending episode experience buffer to whole experience buffer
        diff = eb_size + epi_eb_size - EXPERIENCE_BUFFER_MAX_SIZE
        if diff <= 0:
            eb[eb_size : eb_size + epi_eb_size] = epi_eb[:epi_eb_size]
        else:            
            eb = np.r_[eb[diff : eb_size], epi_eb[:epi_eb_size]]
        eb_size = min(eb_size + epi_eb_size, EXPERIENCE_BUFFER_MAX_SIZE)
        print(f"[experience size: {eb_size}]")
        # progress of some observations
        rewards_ema = rewards_ema * LEARNING_QUALITY_OBSERVATIONS_EMAS_DECAY + rewards_total / rewards_count * (1.0 - LEARNING_QUALITY_OBSERVATIONS_EMAS_DECAY)
        distances_ema = distances_ema * LEARNING_QUALITY_OBSERVATIONS_EMAS_DECAY + distances_total / max_frames * (1.0 - LEARNING_QUALITY_OBSERVATIONS_EMAS_DECAY)
        parked_frequency_ema = parked_frequency_ema * LEARNING_QUALITY_OBSERVATIONS_EMAS_DECAY + car.parked_ * (1.0 - LEARNING_QUALITY_OBSERVATIONS_EMAS_DECAY)
        print(f"[parked frequency: {parked_frequency}]")
        print(f"[parked frequency moving average: {parked_frequency_ema}]")        
        print(f"[rewards moving average: {rewards_ema}]")
        print(f"[distances moving average: {distances_ema}]")        
        extras["parked_count"].append(parked_count)
        extras["parked_frequency"].append(parked_frequency)
        extras["parked_frequency_ema"].append(parked_frequency_ema)
        extras["rewards_ema"].append(rewards_ema)
        extras["distances_ema"].append(distances_ema)
        # learning                    
        if LEARNING_ON and (epi + 1) % QL_FIT_GAP_EPISODES == 0 and (epi + 1) >= QL_FIRST_FIT_AT_EPISODE:
            print(f"[fitting Q...]")
            t1_fit = time.time()            
            m = QL_FIT_BATCH_SIZE                                           
            print(f"[drawing batch...; eb_size: {eb_size}, batch size: {m}]")            
            t1_batch = time.time()
            # prioritized experience replay (below, inactive now)
            # eb_range = np.arange(eb_size)
            # priorities = 1 + np.argsort(np.argsort(experience_buffer[eb_range, 5])) # Bellman errors' ranks as priorities
            # priorities = eb[eb_range, 5].astype(np.float64) # Bellman errors as priorities
            # priorities += 1e-6
            # p = priorities / np.sum(priorities)
            # indexes = np.random.choice(eb_range, m, p=p) # prioritized experience replay, drawing a random batch according to p distribution
            indexes = np.random.choice(eb_size, m) # uniform experience replay            
            t2_batch = time.time()
            print(f"[drawing batch done; time: {t2_batch - t1_batch} s]")            
            # preparing targets
            print(f"[preparing targets on batch...]")
            t1_targets = time.time()            
            eb_batch = eb[indexes]
            eb_batch_range = np.arange(m)
            states = np.array(list(eb_batch[:, 0]))          
            X_batch = QL_TRANSFORMER.fit_transform(states)
            # preliminary vectors of targets for all actions - its particular positions (related to actions taken) shall be prepared using Bellman equation 
            qs = np.zeros((m, len(ACTION_PAIRS)), dtype=np.float64)            
            if False and first_fit_done: # change first condition to False for non-shared architecture of Q-model (preliminary predict unnecessary then)
                qs = Q.predict(X_batch)                
            y_batch = np.copy(qs)                                                    
            next_states = np.array(list(eb_batch[:, 3]))
            X_batch_next = QL_TRANSFORMER.fit_transform(next_states)
            qns_oracle = np.zeros((m, len(ACTION_PAIRS)))
            if Q_oracle is not None:
                qns_oracle = Q_oracle.predict(X_batch_next)    
            qns = np.zeros((m, len(ACTION_PAIRS)))                        
            if first_fit_done:
                qns = Q.predict(X_batch_next)                                                            
            actions_batch = eb_batch[:, 1].astype(int)
            rewards_batch = eb_batch[:, 2].astype(np.float64)
            terminals_batch = np.where(eb_batch[:, 4] == 1)[0]
            rewards_on_terminals_batch = np.array([rewards_batch[terminals_batch]]).T
            qns[terminals_batch, :] = rewards_on_terminals_batch # "virtual" next states preserve rewards of terminals until end of episode (for discounted next max in Bellman equation) 
            qns_oracle[terminals_batch, :] = rewards_on_terminals_batch # "virtual" next states preserve rewards of terminals until end of episode (for discounted next max in Bellman equation)
            qns_argmaxes = np.argmax(qns, axis=1)      
            qns_maxes = qns_oracle[eb_batch_range, qns_argmaxes]                                                                        
            y = rewards_batch + QL_GAMMA * qns_maxes
            y_batch[eb_batch_range, actions_batch] = y                        
            y_pred = np.zeros((m, len(ACTION_PAIRS)))
            if first_fit_done:
                y_pred = Q.predict(X_batch)
            maes_batch_before = np.abs(y_pred[eb_batch_range, actions_batch] - y_batch[eb_batch_range, actions_batch])            
            mse_batch_before = np.mean(maes_batch_before**2)
            sse_batch_before = np.sum(maes_batch_before**2)
            v_batch = np.sum((y_batch[eb_batch_range, actions_batch] - np.mean(y_batch[eb_batch_range, actions_batch]))**2)
            fvu_batch_before = sse_batch_before / v_batch
            r2_batch_before = 1.0 - fvu_batch_before
            # prioritized experience replay (below, inactive now)
            # u_indexes = np.unique(indexes)
            # for u_index in u_indexes:
            #     u_index_where = np.where(u_indexes == u_index)[0][0]                    
            #     eb[u_index, 5] = maes_batch_before[u_index_where]            
            t2_targets = time.time()
            print(f"[preparing targets on batch done; time: {t2_targets - t1_targets} s]")
            print(f"[mse on batch before fit: {mse_batch_before}]")
            print(f"[r^2 on batch before fit: {r2_batch_before}]")            
            # Q fit
            print(f"[actual fit...]") 
            t1_actual_fit = time.time()                                     
            if Q is None and Q_oracle is None:                    
                Q = QL_APPROXIMATOR
                Q_oracle = None
            Q.fit(X_batch, y_batch, actions_batch)                        
            first_fit_done = True            
            y_pred = Q.predict(X_batch)
            maes_batch_after = np.abs(y_pred[eb_batch_range, actions_batch] - y_batch[eb_batch_range, actions_batch])                                
            mse_batch_after = np.mean(maes_batch_after**2)
            sse_batch_after = np.sum(maes_batch_after**2)            
            fvu_batch_after = sse_batch_after / v_batch
            r2_batch_after = 1.0 - fvu_batch_after                                                                                            
            t2_actual_fit = time.time()
            print(f"[actual fit done; time: {t2_actual_fit - t1_actual_fit} s]") 
            print(f"[mse on batch after fit: {mse_batch_after}, mse diff: {mse_batch_before - mse_batch_after}]")
            print(f"[r^2 on batch after fit: {r2_batch_after}, r^2 diff: {r2_batch_after - r2_batch_before}]")
            extras["mse_batch_before"].append(mse_batch_before)
            extras["mse_batch_after"].append(mse_batch_after)          
            extras["r2_batch_before"].append(r2_batch_before)
            extras["r2_batch_after"].append(r2_batch_after)                       
            t2_fit = time.time()
            print(f"[fitting Q done; total fit time: {t2_fit - t1_fit} s]")
            if QL_ORACLE_SLOW_UPDATES_DECAY < 1.0 and Q_oracle is not None:
                print("[slow Q_target update...]") 
                Q_oracle.average_with_other(Q, 1.0 - QL_ORACLE_SLOW_UPDATES_DECAY)
                print("[slow Q_target update done.]")
        if LEARNING_ON and QL_ORACLE_SWITCHING and (epi + 1)  % QL_ORACLE_SWITCH_GAP_EPISODES == 0 and (epi + 1) >= QL_ORACLE_FIRST_SWITCH_AFTER_EPISODE:
            if Q_oracle is None or not QL_ORACLE_SLOW_UPDATES_DECAY < 1.0:
                print("[switching Q_target...]")         
                Q_oracle = deepcopy(Q)
                print("[switching Q_target done.]")
        eps = max(QL_EPS_MIN, QL_EPS_MAX - (epi + 1) / (QL_EPS_MIN_AT_EPISODE - 1) * (QL_EPS_MAX - QL_EPS_MIN)) 
        t2_loop_body = time.time()            
        print(f"[whole loop body time: {t2_loop_body - t1_loop_body} s]")
    if LEARNING_ON:    
        pickle_all(FOLDER_MODELS + f"{ehs}_q.bin", [Q])
        Q.json_dump(FOLDER_MODELS + f"{ehs}_q.json")
        pickle_all(FOLDER_EXTRAS + f"{ehs}_extras.bin", [extras])           
    t2_main = time.time()
    print(f"CAR PARKING EXPERIMENT DONE. [time: {t2_main - t1_main} s]")