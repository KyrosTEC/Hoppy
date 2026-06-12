import mujoco, mujoco.viewer
import numpy as np, math, os, time
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ══ MODE ══════════════════════════════════════════════════════
ORBITAL = True   # True = circles  |  False = in-place
# ═════════════════════════════════════════════════════════════

M4, LK = 0.149, 0.1545
TAU_KNEE_MAX  = 4.67
TAU_HIP_MAX   = 4.36
TAU_PITCH_MAX = 50.0
TAU_YAW_MAX   = 150.0   # must overcome 112 Nm coupling force

J2_IC        =  0.244
J2_COMPRESS  =  0.08         # arm compression depth
Q3_IC        =  math.pi/3
Q4_IC        = -math.pi/2
Q4_STANCE_MAX=  0.0

Tst = 0.20

KP_PITCH = 200; KD_PITCH = 20
KP_HIP   =  50; KD_HIP   =  3
KP_KNEE  = 150; KD_KNEE  =  8

YAW_SPEED  = 0.8     # orbital rad/s  (1 circle ≈ 7.9 s)
KP_YAW_ORB = 150; KI_YAW_ORB = 20   # high gain to overcome coupling
KP_YAW_LCK =  50; KD_YAW_LCK = 10  # in-place lock

DT = 0.001; LAM = 10.0


def spring_tau(q4):
    return 2.0*(-0.0242*q4 + 0.0108)

def grav_comp_knee(q2, q3, q4):
    return 9.81 * math.cos(q2) * M4 * LK * math.sin(q3 + q4)


class VelFilt:
    def __init__(self): self._p = self._f = 0.0
    def reset(self, p0=0.0): self._p = p0; self._f = 0.0
    def update(self, q):
        self._f = (1-LAM*DT)*self._f + LAM*DT*(q-self._p)/DT
        self._p = q; return self._f


class HoppyController:
    def __init__(self):
        self.state   = "STANCE"
        self.t_td    = 0.0
        self._yaw_ei = 0.0
        self._fp = VelFilt(); self._fh = VelFilt(); self._fk = VelFilt()
        self.log = {k: [] for k in
                    ['t','q2','q3','q4','tau_p','tau_h','tau_k',
                     'state','foot_z','touch','yaw_vel']}

    def step(self, model, data):
        t  = data.time
        q1 = data.qpos[0]
        q2 = data.qpos[1]; q3 = data.qpos[2]; q4 = data.qpos[3]
        dq1 = data.qvel[0]
        dq2 = self._fp.update(q2)
        dq3 = self._fh.update(q3)
        dq4 = self._fk.update(q4)

        touch = float(data.sensor('foot_touch').data[0])
        fz    = float(data.sensor('foot_pos').data[2])

        if self.state == "STANCE":
            if (t - self.t_td) >= Tst: self.state = "FLIGHT"
        else:
            if touch > 0.5: self.state = "STANCE"; self.t_td = t

        s   = float(np.clip((t - self.t_td) / Tst, 0, 1))
        env = math.sin(math.pi * s)

        # Pitch: compress in stance, release in flight
        j2_target = J2_COMPRESS if self.state == "STANCE" else J2_IC
        tP = float(np.clip(
            KP_PITCH*(j2_target - q2) - KD_PITCH*dq2,
            -TAU_PITCH_MAX, TAU_PITCH_MAX))

        tH = float(np.clip(
            KP_HIP*(Q3_IC - q3) - KD_HIP*dq3,
            -TAU_HIP_MAX, TAU_HIP_MAX))

        Ge = grav_comp_knee(q2, q3, q4)
        ts = spring_tau(q4)
        q4_tgt = Q4_IC + (Q4_STANCE_MAX - Q4_IC)*env if self.state == "STANCE" else Q4_IC
        tK = float(np.clip(
            KP_KNEE*(q4_tgt - q4) - KD_KNEE*dq4 + Ge + ts,
            -TAU_KNEE_MAX, TAU_KNEE_MAX))

        data.ctrl[0] = tP; data.ctrl[1] = tH; data.ctrl[2] = tK

        # Yaw
        if ORBITAL:
            yaw_err      = YAW_SPEED - dq1
            self._yaw_ei = float(np.clip(self._yaw_ei + yaw_err*DT, -3, 3))
            tY = float(np.clip(
                KP_YAW_ORB*yaw_err + KI_YAW_ORB*self._yaw_ei,
                -TAU_YAW_MAX, TAU_YAW_MAX))
        else:
            tY = float(np.clip(
                KP_YAW_LCK*(0.0 - q1) - KD_YAW_LCK*dq1,
                -TAU_YAW_MAX, TAU_YAW_MAX))
        data.ctrl[3] = tY

        for k, v in zip(self.log.keys(), [
            t, q2, q3, q4, tP, tH, tK,
            float(self.state == "STANCE"), fz, touch, dq1
        ]):
            self.log[k].append(v)


def plot_results(log, path):
    t   = np.array(log['t'])
    sta = np.array(log['state'])
    td  = int(np.sum(np.diff(sta.astype(int)) > 0))
    fzv = np.array(log['foot_z'])
    kv  = np.degrees(np.array(log['q4']))
    yv  = np.array(log['yaw_vel'])

    def shade(ax):
        d_ = np.diff(sta.astype(int))
        for s in np.where(d_>0)[0]:
            ends = np.where(d_<0)[0]
            e = ends[ends>s][0] if len(ends[ends>s]) else len(t)-1
            ax.axvspan(t[s], t[e], alpha=0.15, color='#e74c3c')

    mode_str = "ORBITAL" if ORBITAL else "IN-PLACE"
    fig, axes = plt.subplots(4,1, figsize=(14,14), sharex=True)
    fig.suptitle(
        f"HOPPY — {mode_str} | compress {J2_IC*1000:.0f}mm→{J2_COMPRESS*1000:.0f}mm\n"
        f"knee {kv.min():.0f}°→{kv.max():.0f}° | "
        f"{td} TDs = {td/max(t[-1],1):.1f} Hz | "
        f"foot max = {fzv.max()*100:.1f} cm",
        fontsize=11, fontweight='bold')

    def panel(ax, ys, lbls, cols, ylabel, title, hlines=None):
        for y,l,c in zip(ys,lbls,cols): ax.plot(t, y, c, lw=1.3, label=l)
        if hlines:
            for v,c,l in hlines: ax.axhline(v, color=c, ls='--', lw=0.9, label=l)
        shade(ax); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(ncol=4, fontsize=8); ax.grid(alpha=0.3)

    panel(axes[0], [np.array(log['q2'])*1000],
          ['arm j2 [mm]'], ['#e67e22'], '[mm]', 'Arm height',
          hlines=[(J2_IC*1000,'#888','IC'), (J2_COMPRESS*1000,'#e74c3c','compress')])
    panel(axes[1], [fzv*100], ['foot_z [cm]'], ['#27ae60'], '[cm]',
          'Foot height (red=STANCE)', hlines=[(1.8,'#888','floor')])
    panel(axes[2], [kv, np.degrees(log['q3'])],
          ['knee [°]','hip [°]'], ['#e74c3c','#2980b9'], '[°]',
          'Joint angles — knee natural range',
          hlines=[(-90,'#e74c3c','-90°'),(0,'#e74c3c','0°'),(60,'#2980b9','60°')])
    panel(axes[3], [yv], ['yaw vel [rad/s]'], ['#8e44ad'], '[rad/s]',
          'Yaw — orbital or locked',
          hlines=[(YAW_SPEED if ORBITAL else 0, '#555',
                   f'target={YAW_SPEED}' if ORBITAL else '0')])
    axes[3].set_xlabel('Time [s]')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"[OK] Plots → {path}")
    plt.close()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(here, "hoppy.xml")
    mode_str = "ORBITAL (circles)" if ORBITAL else "IN-PLACE (locked)"

    print("=" * 60)
    print(f"  HOPPY — {mode_str}")
    print("=" * 60)
    print(f"  Arm compress: {J2_IC*1000:.0f}mm → {J2_COMPRESS*1000:.0f}mm")
    print(f"  Knee range  : -90° → 0°  (natural)")
    print(f"  Yaw         : {'orbit ' + str(YAW_SPEED) + ' rad/s (~8s/circle)' if ORBITAL else 'locked at 0°'}")
    print(f"  Toggle      : set ORBITAL = True/False at top of file")
    print("=" * 60)

    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)
    data.qpos[0] = 0.0; data.qpos[1] = J2_IC
    data.qpos[2] = Q3_IC; data.qpos[3] = Q4_IC
    data.qvel[0] = YAW_SPEED if ORBITAL else 0.0
    mujoco.mj_forward(model, data)

    ctrl = HoppyController()
    ctrl._fp.reset(J2_IC); ctrl._fh.reset(Q3_IC); ctrl._fk.reset(Q4_IC)

    foot_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'foot')
    r = model.geom_size[foot_geom, 0]
    print(f"[INFO] foot = {(data.geom_xpos[foot_geom][2]-r)*1e3:.1f}mm  ncon={data.ncon}")

    SIM_DURATION = 15.0
    print(f"[INFO] Simulating {SIM_DURATION}s …")

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Good camera for seeing the full arm + leg motion
            viewer.cam.distance  = 1.8
            viewer.cam.azimuth   = 155
            viewer.cam.elevation = -18
            viewer.cam.lookat[:] = [-0.25, 0.0, 0.18]
            while viewer.is_running() and data.time < SIM_DURATION:
                t0 = time.time()
                ctrl.step(model, data); mujoco.mj_step(model, data)
                viewer.sync()
                slack = DT - (time.time()-t0)
                if slack > 0: time.sleep(slack)
    except Exception as e:
        print(f"[Viewer] {e} — headless")
        while data.time < SIM_DURATION:
            ctrl.step(model, data); mujoco.mj_step(model, data)

    sta = np.array(ctrl.log['state'])
    td  = int(np.sum(np.diff(sta.astype(int)) > 0))
    fzv = np.array(ctrl.log['foot_z'])
    kv  = np.degrees(np.array(ctrl.log['q4']))
    yv  = np.array(ctrl.log['yaw_vel'])
    circles = data.qpos[0] / (2*math.pi) if ORBITAL else 0

    print(f"\n[OK] Mode       : {mode_str}")
    print(f"[OK] Touchdowns : {td}  ({td/SIM_DURATION:.1f} Hz)")
    print(f"[OK] foot_z max : {fzv.max()*100:.1f} cm")
    print(f"[OK] knee range : {kv.min():.1f}° → {kv.max():.1f}°")
    if ORBITAL:
        print(f"[OK] Circles    : {circles:.2f}  ({circles/SIM_DURATION*60:.1f} circles/min)")
        print(f"[OK] Yaw final  : {yv[-1]:.3f} rad/s")

    plot_results(ctrl.log, os.path.join(here, "hoppy_results.png"))


if __name__ == "__main__":
    main()