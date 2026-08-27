import os
import glob
import imageio.v2 as imageio
import numpy as np
from PIL import Image

OUTPUT_FORMAT = 'gif'
INPUT_FILENAME = '/scratch/gb02/pc2687/am3-plots/*Jan-*.png'
OUTPUT_FILENAME = './animation'
FPS = 6
MP4_QUALITY = 26 # lower is better, range 0 - 51
GIF_FRAME_STEP = 1 # use every nth input frame
GIF_SCALE = 0.5 # resize GIF frames; e.g. 0.5 halves width and height
GIF_PALETTE_SIZE = 256 # fewer colors reduce size; range 2 - 256

def main():
    if OUTPUT_FORMAT == 'gif':
        return make_gif(
            INPUT_FILENAME,
            OUTPUT_FILENAME,
            fps=FPS,
            frame_step=GIF_FRAME_STEP,
            scale=GIF_SCALE,
            palettesize=GIF_PALETTE_SIZE,
        )

    elif OUTPUT_FORMAT == 'mp4':
        make_mp4(INPUT_FILENAME, OUTPUT_FILENAME, fps=FPS, quality=MP4_QUALITY)
        
    else:
        raise ValueError(f'Unknown output format: {OUTPUT_FORMAT}, use gif or mp4')

    return 


def make_gif(fnamein, fnameout, fps=12, frame_step=1, scale=1.0, palettesize=256):
    '''Create a GIF from image files matching fnamein.'''
    fnames = sorted(glob.glob(fnamein))
    if not fnames:
        raise FileNotFoundError(f'No input frames matched: {fnamein}')
    if frame_step < 1:
        raise ValueError('frame_step must be at least 1')
    if not 0 < scale <= 1:
        raise ValueError('scale must be greater than 0 and no greater than 1')
    if not 2 <= palettesize <= 256:
        raise ValueError('palettesize must be between 2 and 256')

    duration = frame_step / fps
    with imageio.get_writer(
        f'{fnameout}.gif',
        mode='I',
        duration=duration,
        loop=0,
        palettesize=palettesize,
    ) as writer:
        for fname in fnames[::frame_step]:
            frame = imageio.imread(fname)
            if scale != 1.0:
                new_size = (int(frame.shape[1] * scale), int(frame.shape[0] * scale))
                frame = np.asarray(Image.fromarray(frame).resize(new_size, Image.Resampling.LANCZOS))
            writer.append_data(frame)

    return f'completed, see: {fnameout}.gif'

def make_mp4(fnamein,fnameout,fps=12,quality=26):
    '''
    Uses ffmpeg to create mp4 with custom codec and options for maximum compatability across OS.
        fnamein (string): The image files to create animation from, with glob wildcards (*) accepted.
        fnameout (string): The output filename (excluding extension)
        fps (float): The frames per second. Default 12.
        quality (float): quality ranges 0 to 51, 51 being worst.
    '''

    # collect animation frames
    fnames = sorted(glob.glob(fnamein))
    if not fnames:
        raise FileNotFoundError(f'No input frames matched: {fnamein}')

    img_shp = imageio.imread(fnames[0]).shape
    out_h, out_w = img_shp[0],img_shp[1]

    # resize output to blocksize for maximum capatability between different OS
    macro_block_size=16
    if out_h % macro_block_size > 0:
        out_h += macro_block_size - (out_h % macro_block_size)
    if out_w % macro_block_size > 0:
        out_w += macro_block_size - (out_w % macro_block_size)

    # quality ranges 0 to 51, 51 being worst.
    assert 0 <= quality <= 51, "quality must be between 1 and 51 inclusive"

    # use ffmpeg command to create mp4
    command = f'ffmpeg -framerate {fps} -pattern_type glob -i "{fnamein}" \
        -vcodec libx264 -crf {quality} -s {out_w}x{out_h} -pix_fmt yuv420p -y {fnameout}.mp4'
    os.system(command)

    return f'completed, see: {fnameout}.mp4'

if __name__ == "__main__":
    main()

