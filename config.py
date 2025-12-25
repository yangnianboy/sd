DEVICE = 'cuda'

MODEL_STAGE_1 = "DeepFloyd/IF-I-L-v1.0"
MODEL_STAGE_2 = "DeepFloyd/IF-II-L-v1.0"
MODEL_STAGE_3 = "stabilityai/stable-diffusion-x4-upscaler"

DEFAULT_PROMPTS = [
    'painting of a snowy mountain village',
    'painting of a horse'
]

VIEW_OPTIONS = [
    'identity',
    'rotate_180',
    'rotate_cw',
    'rotate_ccw',
    'flip',
    'negate',
    'skew',
    'patch_permute',
    'pixel_permute',
    'inner_circle',
    'square_hinge',
    'jigsaw'
]

DEFAULT_VIEW = ['identity', 'rotate_cw']

NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 10.0
NOISE_LEVEL = 50

