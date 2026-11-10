# Third-party materials

| Material | Location | License/source record |
| --- | --- | --- |
| PX4 sources and existing SITL runtime | `third_party/PX4-Autopilot/`, `runtime/px4/` | PX4 BSD 3-Clause; `licenses/PX4-BSD-3-Clause.txt`; nested components retain their own notices |
| pymavlink 2.4.47 source snapshot | `src/pymavlink/` | Existing `COPYING`, module copyright headers and `licenses/pymavlink-COPYING.txt` |
| RotorPy 2.1.2 Hummingbird parameter file | `src/wind_field_simulation/_vendor/hummingbird_params.py` | MIT; `licenses/RotorPy-MIT.txt` |
| Existing PINN model and dataset/normalizer code | `src/wind_field_simulation/_vendor/pinn.py`, `dataset.py` | Copied from local PINN project; no top-level license found there |
| PINN model and normalizer weights | `src/wind_field_simulation/assets/checkpoints/` | Copied existing research artifacts; redistribution terms need owner selection |
| X4-950 and Airbot model/scene assets | `src/wind_field_simulation/assets/` | Existing headers and notices preserved, including RAIDS Lab all-rights-reserved XML headers |
| Cabinet scene | `assets/robot_models/scenes/cabinet/` inside the package | Existing `LICENSE.source` preserved |

The migrated application retains its existing behavior. Bundled dependency code
is preserved for compatibility; it has not been rewritten to apply new style
conventions or retrained. No blanket open-source license is assigned by this
migration. Before public release, the copyright holders should choose terms for
their application code, models and weights and retain all applicable notices.

The wheel includes application resources and Python dependency snapshots. The
first GitHub version ships the host-specific PX4 SITL executable and startup
resources, documentation and license records in the project directory, not the
Python wheel. The complete local PX4 source tree is retained locally but omitted
from this first upload; source baselines and adaptation patches remain recorded
under `docs/provenance/`. No third-party licensing notices are removed.
