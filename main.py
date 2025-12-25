import argparse
import mediapy as mp
from PIL import Image
import numpy as np

from generate import VisualAnagramGenerator
from config import DEFAULT_PROMPTS, VIEW_OPTIONS, DEFAULT_VIEW

def parse_args():
    parser = argparse.ArgumentParser(description='Visual Anagrams Generator')
    parser.add_argument('--token', type=str, required=True, help='HuggingFace token')
    parser.add_argument('--prompt1', type=str, default=DEFAULT_PROMPTS[0])
    parser.add_argument('--prompt2', type=str, default=DEFAULT_PROMPTS[1])
    parser.add_argument('--view', type=str, default=DEFAULT_VIEW[1], choices=VIEW_OPTIONS[1:])
    parser.add_argument('--output', type=str, default='./output')
    parser.add_argument('--animate', action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()
    
    generator = VisualAnagramGenerator(args.token)
    prompts = [args.prompt1, args.prompt2]
    views = ['identity', args.view]
    
    print(f"Prompts: {prompts}")
    print(f"Views: {views}")
    
    results = generator.generate(prompts, views)
    
    for size, image in results.items():
        view_images = generator.get_view_images(image)
        for i, img in enumerate(view_images):
            Image.fromarray(img).save(f'{args.output}_{size}_view{i}.png')
    
    print(f"Images saved to {args.output}_*.png")
    
    if args.animate:
        video_path = generator.save_animation(results['1024'], f'{args.output}_animation.mp4')
        print(f"Animation saved to {video_path}")

if __name__ == '__main__':
    main()

