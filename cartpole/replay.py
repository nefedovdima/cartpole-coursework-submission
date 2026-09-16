"""Render a saved episode as self-contained HTML and a time-correct VFR MP4."""
import argparse
import base64
from datetime import datetime, timezone
from fractions import Fraction
import html
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import matplotlib
matplotlib.use('Agg')

from cartpole.common.episode_log import load_log
from cartpole.common.view import EpisodeScene, replay_frame_times


def export_replay(log_path, output, fps=25):
    log = load_log(log_path)
    if log['completion']['reason'] == 'exception':
        raise ValueError('an incomplete episode with an exception cannot be presented as a completed replay')
    physical_start = log['states'][0]['time']
    # Quantize absolute times once. Rounding each floating-point interval
    # independently would accumulate timing drift across a long replay.
    samples = {round((stamp-physical_start)*1000000): stamp
               for stamp in replay_frame_times(log, fps)}
    ticks, times = list(samples), list(samples.values())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    scene = EpisodeScene(log)
    previews = {'frame_start.png': 0, 'frame_mid.png': len(times)//2, 'frame_end.png': len(times)-1}
    try:
        with tempfile.TemporaryDirectory(prefix='frames-', dir=output) as temporary:
            frames = Path(temporary)
            entries = ['ffconcat version 1.0']
            for index, timestamp in enumerate(times):
                filename = f'{index:06d}.png'
                scene.draw(timestamp)
                scene.figure.savefig(frames/filename, dpi=100, facecolor=scene.figure.get_facecolor())
                for preview, frame_index in previews.items():
                    if index == frame_index:
                        shutil.copyfile(frames/filename, output/preview)
                entries += [f"file '{filename}'", 'option framerate 1000000']
                if index+1 < len(times):
                    entries.append(f'duration {(ticks[index+1]-ticks[index])/1000000:.6f}')
            manifest = frames/'frames.ffconcat'
            manifest.write_text('\n'.join(entries)+'\n')
            nominal_rate = Fraction(str(fps)).limit_denominator(1000000)
            hold_ticks = max(1, round(1000000/fps))
            # Input timestamps retain microsecond resolution. The codec's
            # nominal rate is only a hint; forcing output -r would quantize
            # a shortened last interval. Set only the final packet's duration
            # to a one-frame still hold, without adding a simulated step.
            command = ['ffmpeg', '-v', 'error', '-n', '-f', 'concat', '-safe', '0', '-i', str(manifest),
                       '-c:v', 'libx264', '-preset', 'medium', '-crf', '20', '-pix_fmt', 'yuv420p', '-bf', '0',
                       '-fps_mode', 'vfr', '-enc_time_base', '1:1000000',
                       '-x264-params', f'fps={nominal_rate.numerator}/{nominal_rate.denominator}',
                       '-bsf:v', f"setts=duration='if(eq(N,{len(times)-1}),{hold_ticks},DURATION)'",
                       '-video_track_timescale', '1000000', '-movflags', '+faststart', str(output/'animation.mp4')]
            try:
                subprocess.run(command, check=True, capture_output=True)
            except subprocess.CalledProcessError as exc:
                exc.add_note(exc.stderr.decode(errors='replace'))
                raise
    finally:
        scene.close()
    video = base64.b64encode((output/'animation.mp4').read_bytes()).decode('ascii')
    poster = base64.b64encode((output/'frame_start.png').read_bytes()).decode('ascii')
    local_hold = log['metadata'].get('experiment') == 'local_upright_lqr_hold'
    swing_up = log['metadata'].get('experiment') == 'classical_swing_up'
    title = html.escape((log['metadata']['controller_label']+' · ' if 'controller_label' in log['metadata']
                        else 'LQR · локальное удержание сверху · ' if local_hold
                         else 'Классический подъём и LQR · ' if swing_up else 'Сохранённый эпизод · ')
                        + log['metadata'].get('case', 'episode'))
    purpose = 'Это локальное удержание, без подъёма снизу.' if local_hold else 'Состояния и управление взяты из сохранённого лога.'
    document = f'''<!doctype html>
<html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>body{{margin:24px auto;max-width:1120px;padding:0 16px;background:#f7f9fc;color:#25344a;font:16px system-ui,sans-serif}}video{{width:100%;border-radius:12px}}h1{{font-size:22px}}p{{line-height:1.5}}</style>
<h1>{title}</h1>
<video controls playsinline preload="metadata" poster="data:image/png;base64,{poster}" src="data:video/mp4;base64,{video}"></video>
<p>Воспроизведение сохранённого эпизода: {physical_start:.6f}–{times[-1]:.6f} с. После завершения конечный кадр остаётся неподвижным ещё на {1/fps:.3f} с. {purpose}</p>
</html>'''
    (output/'animation.html').write_text(document, encoding='utf-8')
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                                               '-show_entries', 'stream=codec_name,width,height,avg_frame_rate,nb_frames,duration:format=duration',
                                               '-of', 'json', str(output/'animation.mp4')]))
    description = {'source_log': str(Path(log_path).resolve()), 'fps_nominal': fps,
                   'physical_frame_times': times, 'encoded_frame_times': [tick/1000000 for tick in ticks],
                   'physical_start_time': physical_start, 'physical_end_time': times[-1],
                   'final_frame_hold_seconds': hold_ticks/1000000, 'timestamp_resolution_seconds': 1e-6,
                   'state_interpolation': 'linear interpolation of saved unwrapped coordinates; held transition input',
                   'ffprobe': probe, 'manual_interactive_playback_checked': False}
    (output/'replay_metadata.json').write_text(json.dumps(description, indent=2)+'\n')
    return output


def main():
    parser = argparse.ArgumentParser(description='HTML/MP4 из сохранённого JSON-лога, без расчёта динамики.')
    parser.add_argument('log', type=Path)
    parser.add_argument('--output-dir', type=Path, help='Новый каталог экспорта, без перезаписи.')
    parser.add_argument('--fps', type=float, default=25.)
    args = parser.parse_args()
    output = args.output_dir or args.log.parent/(args.log.stem+datetime.now(timezone.utc).strftime('_replay_%Y%m%dT%H%M%S_%fZ'))
    print(export_replay(args.log, output, args.fps).resolve())


if __name__ == '__main__':
    main()
