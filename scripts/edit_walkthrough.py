"""Reproduce the published walkthrough crop from the author's original recording.

Requires FFmpeg with libx264 and drawtext. Original recording is not distributed.
The edit preserves playback speed and uses the original displayed decoder labels
for switch timing. No model output is regenerated or enhanced.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

SOURCE_SHA256 = '1226e7c44960555211fd7895ef3a76a03b8cf685c6b3e0c3af9107eac3236db2'
START_FRAME, END_FRAME = 480, 1950  # 8.0 to 32.5 s at 60 fps
SWITCHES = [(999, 'B', 'circular128'), (1423, 'A', 'native'),
            (1666, 'B', 'circular128')]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, default=Path('docs/media/h3-seam-walkthrough.mp4'))
    parser.add_argument('--font', type=Path,
                        default=Path('/usr/share/fonts/liberation/LiberationSans-Regular.ttf'))
    args = parser.parse_args()
    with args.source.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    if digest != SOURCE_SHA256:
        parser.error('Source hash differs from the recording used for this edit')
    if not args.font.is_file():
        parser.error('Supply an installed TrueType font with --font')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Escape the filter path; it is not evaluated by a shell.
    font = str(args.font).replace('\\', '\\\\').replace(':', '\\:').replace("'", "'\\''")
    first, second, third = [frame - START_FRAME for frame, *_ in SWITCHES]
    a = f'lt(n,{first})+between(n,{second},{third-1})'
    b = f'between(n,{first},{second-1})+gte(n,{third})'
    filters = [
        f'trim=start_frame={START_FRAME}:end_frame={END_FRAME}',
        'setpts=PTS-STARTPTS',
        # Only the moving comparison canvas. Browser UI and old labels are removed.
        'crop=1412:680:244:222',
        'pad=1412:736:0:56:color=0x14181d',
        f"drawtext=fontfile='{font}':text='A  |  Native decode':fontsize=28:fontcolor=white:x=20:y=14:enable='{a}'",
        f"drawtext=fontfile='{font}':text='B  |  Circular decode - 128 px':fontsize=28:fontcolor=white:x=20:y=14:enable='{b}'",
        f"drawtext=fontfile='{font}':text='Same latent - 50 steps':fontsize=20:fontcolor=0xd5dce0:x=w-tw-20:y=18",
    ]
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-y',
                    '-i', str(args.source), '-vf', ','.join(filters), '-an',
                    '-c:v', 'libx264', '-preset', 'slow', '-crf', '18',
                    '-pix_fmt', 'yuv420p', '-fps_mode', 'passthrough',
                    '-movflags', '+faststart', '-map_metadata', '-1',
                    str(args.output)], check=True)
    poster = args.output.with_suffix('.jpg')
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                    '-ss', '12', '-i', str(args.output), '-frames:v', '1',
                    '-q:v', '2', str(poster)], check=True)
    receipt = {
        'source_sha256': digest, 'source_fps': 60,
        'source_frame_range': [START_FRAME, END_FRAME],
        'end_frame_exclusive': True, 'crop_xywh': [244, 222, 1412, 680],
        'caption_header_pixels': 56, 'playback_speed': 1.0,
        'duration_seconds': (END_FRAME-START_FRAME)/60,
        'switches': [{'source_frame': f, 'output_frame': f-START_FRAME,
                      'label': label, 'decoder': decoder}
                     for f, label, decoder in SWITCHES],
        'switch_provenance': 'Displayed comparison labels inspected at source frame boundaries.',
        'output_codec': 'H.264', 'crf': 18, 'pixel_format': 'yuv420p',
        'output_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
        'note': 'Screen recording with moving viewpoint and advancing playback. PDF stills provide fixed-frame comparisons.'
    }
    args.output.with_suffix('.json').write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    main()
