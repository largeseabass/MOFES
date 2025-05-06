import tempfile
from pathlib import Path
from mofes import Cartesian_MosquitoSolver, Neches_problem_settings

def test_spinup_execution():
    # Resolve dummy stag-water path
    this_dir = Path(__file__).resolve().parent
    stag_path = this_dir.parent / "data" / "stag-water"
    assert stag_path.exists(), f"Stag-water path not found: {stag_path}"

    with tempfile.TemporaryDirectory() as tmpdir:
        settings = Neches_problem_settings(
            diffusion_coefficient=100.0,
            mu_1=0.1,
            mu_2=0.01,
            oviation_rate=10,
            hatching_rate=0.1,
            immobile_maturation_rate=0.01,
            carrying_capacity=1.0,
            constant_for_mobile=0.01,
            period=0.5,
            steps=1,
            nx=4,
            ny=4,
            save_path=tmpdir,
            output_name="test_output",
            stag_path=str(stag_path),
            flag_spinup=True,
            flag_advection=False,
            flag_observation=False
        )

        solver = Cartesian_MosquitoSolver(settings)
        solver.spin_up()

        output_path = Path(tmpdir) / "test_output_spin_up.xdmf"
        assert output_path.exists(), "Spin-up output file not found."

def test_mosquito_solver_execution():
    this_dir = Path(__file__).resolve().parent
    stag_path = this_dir.parent / "data" / "stag-water"
    assert stag_path.exists(), "Stag-water path not found."

    with tempfile.TemporaryDirectory() as tmpdir:
        settings = Neches_problem_settings(
            diffusion_coefficient=100.0,
            mu_1=0.1,
            mu_2=0.01,
            oviation_rate=10,
            hatching_rate=0.1,
            immobile_maturation_rate=0.01,
            carrying_capacity=1.0,
            constant_for_mobile=0.01,
            period=0.5,
            steps=1,
            nx=4,
            ny=4,
            save_path=tmpdir,
            output_name="test_output",
            stag_path=str(stag_path),
            flag_spinup=False,
            flag_advection=False,
            flag_observation=False
        )

        solver = Cartesian_MosquitoSolver(settings)
        solver.solve_mosquito()

        output_path = Path(tmpdir) / "test_output.xdmf"
        assert output_path.exists(), "solve_mosquito output file not found."


if __name__ == "__main__":
    print("Running test_spinup_execution...")
    test_spinup_execution()
    print("✓ spin_up passed.\n")

    print("Running test_mosquito_solver_execution...")
    test_mosquito_solver_execution()
    print("✓ solve_mosquito passed.\n")
