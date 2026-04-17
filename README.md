# Pteropus giganteus Wing Simulation
## Comparative Aerodynamic Analysis: Rigid-Link vs Deformable Membrane

Physics-based simulation of Indian Flying Fox (*Pteropus giganteus*, 1.4 kg, 1.35 m wingspan)
comparing rigid two-segment wing model against spring-damper membrane model.
Implemented in PyBullet using Featherstone's articulated body algorithm.

## Key Results

| Metric | Rigid-Link | Membrane | Change | Literature |
|--------|-----------|----------|--------|------------|
| Mean power (W) | 6.45 | 5.81 | −9.9% | 5–12 W |
| L/D ratio | 3.84 | 4.63 | +20.6% | 3.5–6.5 |
| Mean lift (N) | 3.31 | 3.03 | −8.5% | 3.0–5.0 N |
| Mean drag (N) | 0.86 | 0.66 | −23.3% | 0.4–0.9 N |
| Elastic energy (J) | 0 | 2.245 | — | ~2.2 J ✓ |

Elastic energy matches Swartz et al. (1996) experimental value within 2%.

## Installation

```bash
git clone https://github.com/[your-username]/pteropus-wing-simulation
cd pteropus-wing-simulation/codes
pip install -r requirements.txt
```

## Usage

```bash
python run_simulation.py              # full comparison, 10 cycles
python run_simulation.py --mode rigid # rigid model only
python run_simulation.py --gui        # with PyBullet visualiser
python run_simulation.py --plots      # regenerate all figures from CSV
```

## Project Structure

```text
codes/                  Core simulation modules
bat_params.py           Morphology dataclass (Norberg & Rayner 1987)
urdf_generator.py       PyBullet URDF generation
physics_engine.py       PyBullet setup (960 Hz, Featherstone)
pd_controller.py        PD gains + sinusoidal trajectory
aero_model.py           Quasi-steady blade-element aerodynamics
membrane.py             Spring-damper membrane network
simulation.py           Rigid-link simulation loop
membrane_simulation.py  Membrane simulation loop
analysis/               Figure generation scripts
results/                CSV output data
docs/                   Paper draft and technical notes
```

## Wing Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Body mass | 1.4 kg | Specified |
| Wingspan | 1.35 m | Specified |
| Flap frequency | 2.0 Hz | Aldridge 1986 |
| Shoulder amplitude | ±75° | Aldridge 1986 |
| Elbow amplitude | ±55°, 15° lag | Aldridge 1986 |
| Membrane E | 1.5 MPa | Swartz et al. 1996 |
| Membrane thickness | 0.3 mm | Swartz et al. 1996 |

## References

1. Norberg & Rayner (1987). Phil Trans R Soc B 316:335–427. https://royalsocietypublishing.org/doi/10.1098/rstb.1987.0030
2. Swartz et al. (1992). Nature 359:726–729.
3. Swartz et al. (1996). J Zoology 239:357–378.
4. Aldridge (1986). J Zoology 210:539–559.
5. Thomas (1975). J Exp Biol 63:273–293.
6. Tian et al. (2006). Bioinspir Biomim 1:S10.
7. Taylor et al. (2003). Nature 425:707–711.
8. Hedenstrom & Johansson (2015). J Exp Biol 218:653–663.
9. Dickinson et al. (1999). Science 284:1954–1960.

## License

MIT License
