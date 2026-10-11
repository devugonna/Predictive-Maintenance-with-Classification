import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import data_load
import model_selection
import training
import evaluate
import eda


from config import OUTPUTS_DIR


def main():
    OUTPUTS_DIR.mkdir(exist_ok=True)
    data_load.run()
    model_selection.run()
    training.run()
    evaluate.run()
    eda.run()


if __name__ == "__main__":
    main()
