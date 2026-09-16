import matplotlib.pyplot as plt
import matplotlib.animation as animation

import math
from bisect import bisect_right

from matplotlib.patches import Rectangle, Circle, FancyArrowPatch

from cartpole.common import Config, State
from cartpole.common.episode_log import record_state
from cartpole.common.metrics import UprightThresholds, upright_angle_error


def pole_geometry(state, length):
    """Original convention: theta=0 down, theta=pi up, pivot=(x, 0)."""
    x, theta = state.cart_position, state.pole_angle
    return [x, x+math.sin(theta)*length], [0., -math.cos(theta)*length]


def generate_pyplot_animation(
        config,
        trajectory,
        trajectory_expected=None,
        timestamps=None,
        margin=0.1):
    if trajectory_expected is not None:
        assert len(trajectory) == len(trajectory_expected)

    if timestamps is not None:
        assert len(trajectory) == len(timestamps)

    x_lim = config.max_position + config.pole_length + margin
    y_lim = config.pole_length + margin

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_title('CartPole')
    ax.set_xlim(-x_lim, +x_lim)
    ax.set_ylim(-y_lim, +y_lim)
    ax.set_aspect('equal')
    ax.grid(linestyle='--')

    pole, = ax.plot([], [], 'o-', lw=2, c='red')
    pole_expected, = ax.plot([], [], 'o-', lw=4, c='gray', alpha=0.3)
    timestamp_text = ax.text(0.05, 0.9, '', transform=ax.transAxes, fontsize=16)

    def animate(i):
        pole_x, pole_y = pole_geometry(trajectory[i], config.pole_length)
        pole.set_data(pole_x, pole_y)

        if trajectory_expected is not None:
            pole_x, pole_y = pole_geometry(trajectory_expected[i], config.pole_length)
            pole_expected.set_data(pole_x, pole_y)
    
        if timestamps is not None:
            timestamp_text.set_text(f't={timestamps[i]:.3f}s')

        return pole, pole_expected

    anim = animation.FuncAnimation(fig, animate, len(trajectory))
    plt.close(fig)

    return anim


def replay_frame_times(log, fps=25):
    """Uniform display cadence plus the exact final time, even for a short step."""
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('fps must be positive and finite')
    start, end = log['states'][0]['time'], log['states'][-1]['time']
    times = [start+i/fps for i in range(math.ceil((end-start)*fps)) if start+i/fps < end]
    return times + [end]


def sample_saved_log(log, timestamp):
    """Interpolate recorded coordinates only; use the held transition input.

    Duplicate timestamps from zero-time events are resolved to the last state
    at that time. The unswapped, unwrapped theta is interpolated directly.
    """
    rows, transitions = log['states'], log['transitions']
    times = [row['time'] for row in rows]
    if not times[0] <= timestamp <= times[-1]:
        raise ValueError('replay time must be within the saved episode')
    index = bisect_right(times, timestamp)-1
    if index == len(rows)-1:
        return record_state(rows[-1]), transitions[-1] if transitions else None, True
    before, after = rows[index:index+2]
    fraction = (timestamp-before['time'])/(after['time']-before['time'])
    state = State(**{key: before[key]+fraction*(after[key]-before[key])
                     for key in ('cart_position', 'cart_velocity', 'pole_angle', 'pole_angular_velocity')},
                  error=before['error'], cart_acceleration=before['cart_acceleration'])
    return state, transitions[index], False


class EpisodeScene:
    """Matplotlib artists reusing the original viewer's pendulum geometry."""

    def __init__(self, log):
        self.log = log
        self.config = Config(**log['metadata']['config'])
        self.thresholds = UprightThresholds(**log['metadata']['upright_thresholds'])
        self.figure = plt.figure(figsize=(11, 7), dpi=100, facecolor='#f7f9fc')
        title = ('LQR · локальное удержание маятника сверху'
                 if log['metadata'].get('experiment') == 'local_upright_lqr_hold'
                 else 'Классический подъём · слежение и LQR' if log['metadata'].get('experiment') == 'classical_swing_up'
                 else 'Маятник · воспроизведение сохранённого эпизода')
        title = log['metadata'].get('controller_label', title)
        self.figure.suptitle(title, fontsize=17, y=.97)
        ax = self.figure.add_axes([.06, .43, .58, .49])
        self.axis = ax
        span = self.config.max_position + self.config.pole_length + .07
        ax.set(xlim=(-span, span), ylim=(-self.config.pole_length-.065, self.config.pole_length+.065),
               xlabel='Положение шарнира, м')
        ax.set_aspect('equal')
        ax.spines[['left', 'right', 'top']].set_visible(False)
        ax.set_yticks([])
        ax.axhline(-.043, color='#66758c', lw=4, zorder=1)
        ax.axhline(0, color='#d2dae5', lw=1, zorder=0)
        for side in (-1, 1):
            bound = side*self.config.max_position
            ax.axvline(bound, color='#c04a42', linestyle='--', lw=1.5)
            ax.text(bound, self.config.pole_length+.025, f'{bound:+.2f} м',
                    ha='center', va='bottom', color='#9e3933', fontsize=10)
        self.cart = Rectangle((-.04, -.03), .08, .03, facecolor='#3575b9', edgecolor='#19446e', zorder=4)
        ax.add_patch(self.cart)
        self.wheels = [Circle((offset, -.035), .008, facecolor='#293e59', zorder=5) for offset in (-.025, .025)]
        for wheel in self.wheels:
            ax.add_patch(wheel)
        self.pole, = ax.plot([], [], color='#243b64', lw=4, zorder=6)
        self.pivot, = ax.plot([], [], 'o', color='#14213b', markersize=7, zorder=7)
        self.tip, = ax.plot([], [], 'o', color='#df9638', markersize=11, zorder=7)
        self.arrow = FancyArrowPatch((0, -.075), (.05, -.075), arrowstyle='-|>',
                                     mutation_scale=15, lw=2, color='#b77825')
        ax.add_patch(self.arrow)
        self.info = self.figure.text(.68, .88, '', va='top', fontsize=12, linespacing=1.6, color='#25344a')
        self.status = self.figure.text(.68, .575, '', va='top', fontsize=11, weight='bold')
        threshold = self.thresholds
        self.figure.text(.68, .46,
                         f'Около верха: |ошибка угла| ≤ {math.degrees(threshold.angle):g}°\n'
                         f'|ω| ≤ {threshold.angular_velocity:g} рад/с; |v| ≤ {threshold.velocity:g} м/с\n'
                         f'|x| ≤ {threshold.position:g} м', fontsize=9, color='#57647a', linespacing=1.5, va='top')
        self.time_axes = [self.figure.add_axes([.08, .10, .38, .24]),
                          self.figure.add_axes([.57, .10, .38, .24])]
        times = [row['time'] for row in log['states']]
        series = [[1000*row['cart_position'] for row in log['states']],
                  [math.degrees(upright_angle_error(row['pole_angle'])) for row in log['states']]]
        labels = ['Положение тележки, мм', 'Ошибка относительно верха, °']
        self.cursors, self.dots = [], []
        for plot, values, label, minimum_range in zip(self.time_axes, series, labels, (20., 2.)):
            plot.plot(times, values, color='#5b86b4', lw=1.8)
            plot.axhline(0, lw=.8, color='#788697')
            half = max(minimum_range, max(abs(value) for value in values)*1.15)
            plot.set(xlim=(times[0], max(times[-1], times[0]+.04)), ylim=(-half, half),
                     xlabel='Время, с', title=label)
            plot.grid(alpha=.2)
            self.cursors.append(plot.axvline(times[0], color='#c04a42', lw=1))
            self.dots.append(plot.plot([], [], 'o', color='#c04a42', markersize=5)[0])

    def draw(self, timestamp):
        state, transition, completed = sample_saved_log(self.log, timestamp)
        x = state.cart_position
        pole_x, pole_y = pole_geometry(state, self.config.pole_length)
        self.pole.set_data(pole_x, pole_y)
        self.pivot.set_data([x], [0.])
        self.tip.set_data([pole_x[1]], [pole_y[1]])
        self.cart.set_x(x-.04)
        for wheel, offset in zip(self.wheels, (-.025, .025)):
            wheel.center = (x+offset, -.035)
        acceleration = transition['applied_acceleration'] if transition else None
        self.arrow.set_visible(not completed and acceleration is not None and acceleration != 0)
        if acceleration is not None:
            self.arrow.set_positions((x, -.075), (x+.12*acceleration/self.config.max_acceleration, -.075))
        acceleration_text = 'нет (время не продвигалось)' if acceleration is None else f'{acceleration:+.4f} м/с²'
        angle = upright_angle_error(state.pole_angle)
        self.info.set_text(f't = {timestamp:.6f} / {self.log["states"][-1]["time"]:.6f} с\n'
                           f'x = {x:+.5f} м     v = {state.cart_velocity:+.5f} м/с\n'
                           f'Ошибка угла = {math.degrees(angle):+.5f}°\n'
                           f'ω = {state.pole_angular_velocity:+.5f} рад/с\n'
                           f'{"u последнего интервала" if completed else "u интервала"}:\n{acceleration_text}')
        if transition and 'control_phase' in transition:
            phase = {'swing_up': 'подъём (TVLQR)', 'lqr': 'LQR'}[transition['control_phase']]
            self.info.set_text(self.info.get_text() + f'\nФаза: {phase}')
        if completed:
            completion = self.log['completion']
            labels = {'position_limit': 'граница положения', 'velocity_limit': 'граница скорости',
                      'simulator_error': 'ошибка симулятора'}
            status = ('Завершено: горизонт' if completion['reason'] == 'horizon'
                      else 'Остановка:\n' + ', '.join(labels.get(reason, reason) for reason in completion['stop_reasons']))
            color = '#365c89' if completion['reason'] == 'horizon' else '#ad3e36'
            if 'evaluation' in self.log:
                success = self.log['evaluation']['success_episode']
                status += '\n' + ('Успех: удержание ≥ 2 с' if success else 'Критерий успеха не выполнен')
                color = '#26744b' if success else '#ad3e36'
        else:
            status, color = 'Выполнение', '#365c89'
        near = self.thresholds.contains(state)
        self.status.set_text(status + '\nОколо верха: ' + ('ДА' if near else 'НЕТ'))
        self.status.set_color(color if completed else ('#26744b' if near else '#a66a26'))
        for cursor, dot, value in zip(self.cursors, self.dots, (1000*x, math.degrees(angle))):
            cursor.set_xdata([timestamp, timestamp])
            dot.set_data([timestamp], [value])

    def close(self):
        plt.close(self.figure)
